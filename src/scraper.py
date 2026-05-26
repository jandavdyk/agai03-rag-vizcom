"""
src/scraper.py

Scrapes vizcom.com using its sitemap.xml to discover URLs, then fetches
each page, strips noise, and saves clean text to data/raw/.

Uses smart URL prioritization to ensure core pages (solutions, pricing,
about) get scraped before being crowded out by blog posts.

Run from project root:
    python src/scraper.py
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# ---- Configuration ----
SITEMAP_URL = "https://vizcom.com/sitemap.xml"
OUTPUT_DIR = Path("data/raw")
INDEX_FILE = OUTPUT_DIR / "_index.json"
USER_AGENT = "Mozilla/5.0 (Educational RAG project; AGAI-03 student assignment)"
REQUEST_DELAY = 1.0
MAX_PAGES = 40            # bumped from 25 to capture more of the site
MIN_TEXT_LENGTH = 200

# URL patterns that mark a page as "core" (gets priority over blog posts)
PRIORITY_PATTERNS = [
    r"vizcom\.com/?$",      # homepage
    r"/pricing",
    r"/about",
    r"/company",
    r"/team",
    r"/enterprise",
    r"/security",
    r"/careers",
    r"/contact",
    r"/faq",
    r"/help",
    r"/docs",
    r"/solutions/",         # e.g. /solutions/apparel, /solutions/automotive
    r"/features/",
    r"/products/",
    r"/use-cases/",
    r"/integrations/",
    r"/customers/",
    r"/resources/",
]


def fetch_sitemap_urls(sitemap_url: str) -> list:
    """Fetch a sitemap.xml and return all page URLs.
    Handles both regular sitemaps and 'sitemap index' files.
    """
    print(f"Fetching sitemap: {sitemap_url}")
    response = requests.get(
        sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=15
    )
    response.raise_for_status()

    content = re.sub(r'\sxmlns="[^"]+"', "", response.text, count=1)
    root = ET.fromstring(content)

    urls = []

    sitemap_entries = root.findall(".//sitemap")
    if sitemap_entries:
        print(f"  -> Sitemap index with {len(sitemap_entries)} child sitemaps")
        for entry in sitemap_entries:
            loc = entry.find("loc")
            if loc is not None and loc.text:
                child_urls = fetch_sitemap_urls(loc.text.strip())
                urls.extend(child_urls)
                time.sleep(REQUEST_DELAY)
        return urls

    for url_elem in root.findall(".//url"):
        loc = url_elem.find("loc")
        if loc is not None and loc.text:
            urls.append(loc.text.strip())

    print(f"  -> Found {len(urls)} URLs")
    return urls


def prioritize_urls(urls: list, max_pages: int) -> list:
    """Sort URLs so 'core' pages (solutions, pricing, etc.) come BEFORE
    blog posts. Without this, blog URLs crowd out everything else when we
    cap at MAX_PAGES.
    """
    priority = []
    blog = []
    other = []

    for url in urls:
        if any(re.search(pat, url) for pat in PRIORITY_PATTERNS):
            priority.append(url)
        elif "/blog/" in url:
            blog.append(url)
        else:
            other.append(url)

    print(f"  URL breakdown: {len(priority)} priority, "
          f"{len(other)} other, {len(blog)} blog")

    # Order: priority first, then any non-blog 'other', then blog posts
    combined = priority + other + blog
    return combined[:max_pages]


def fetch_and_clean(url: str) -> dict:
    """Fetch one URL and return a dict with title + clean text, or None."""
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=15
        )
        response.raise_for_status()
    except Exception as e:
        print(f"  x Fetch failed: {e}")
        return None

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "noscript", "iframe", "svg",
                     "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    title = (
        soup.title.string.strip()
        if soup.title and soup.title.string
        else "Untitled"
    )

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return {
        "url": url,
        "title": title,
        "text": text,
        "length": len(text),
    }


def url_to_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "homepage.txt"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)
    return f"{safe}.txt"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urls = fetch_sitemap_urls(SITEMAP_URL)
    except Exception as e:
        print(f"Sitemap fetch failed: {e}")
        print("Falling back to manual URL list")
        urls = [
            "https://vizcom.com/",
            "https://vizcom.com/pricing",
            "https://vizcom.com/solutions/apparel",
        ]

    if not urls:
        print("No URLs to scrape. Exiting.")
        return

    urls = prioritize_urls(sorted(set(urls)), MAX_PAGES)
    print(f"\nScraping {len(urls)} URLs (capped at {MAX_PAGES})\n")

    index = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")

        result = fetch_and_clean(url)
        if result is None:
            continue

        if result["length"] < MIN_TEXT_LENGTH:
            print(f"  o Skipping (only {result['length']} chars of text)")
            continue

        filename = url_to_filename(url)
        filepath = OUTPUT_DIR / filename

        header = (
            f"URL: {result['url']}\n"
            f"Title: {result['title']}\n"
            f"---\n\n"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + result["text"])

        print(f"  v Saved {filename} ({result['length']} chars)")

        index.append({
            "url": result["url"],
            "title": result["title"],
            "filename": filename,
            "length": result["length"],
        })

        time.sleep(REQUEST_DELAY)

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE. Scraped {len(index)} pages -> {OUTPUT_DIR}/")
    print(f"Index written to {INDEX_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()