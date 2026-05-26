"""
src/qa_generator.py

Reads each cleaned page from data/raw/, sends it to Groq's Llama 3.3 70B
with a structured prompt that requests high-quality (question, answer)
pairs grounded in the page content, then writes everything to
data/qa_dataset.csv.

Run from project root:
    python src/qa_generator.py
"""

import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError


load_dotenv()

# ---- Configuration ----
RAW_DIR = Path("data/raw")
INDEX_FILE = RAW_DIR / "_index.json"
OUTPUT_CSV = Path("data/qa_dataset.csv")

GROQ_MODEL = "llama-3.3-70b-versatile"   # high quality for Q/A generation
QA_PER_PAGE = 8                          # 8 * 25 pages = 200 pairs
MAX_CHARS = 8000                         # truncate very long pages
REQUEST_DELAY = 2.0                      # be polite to the API


PROMPT_TEMPLATE = """You are an expert at creating high-quality question-answer pairs from documentation.

Below is content from a webpage about Vizcom (an AI design platform for product designers and creative teams).
Generate exactly {n} diverse, useful question-answer pairs based ONLY on this content.

Requirements:
- Questions should be natural things a real user might ask about Vizcom or the topic covered.
- Answers MUST be grounded in the content. Do not invent facts.
- Answers should be 1-3 sentences, complete but concise.
- Mix question types: factual ("What is..."), how-to ("How do I..."), comparison ("What's the difference between..."), capability ("Can Vizcom..."), and conceptual ("Why does...").
- Skip questions you cannot fully answer from the content.

Page URL: {url}
Page Title: {title}

Content:
---
{content}
---

Return ONLY valid JSON in this exact format with no other text before or after:
{{
  "pairs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}
"""


def generate_qa_for_page(client, page_info, content):
    """Call Groq to generate Q/A pairs for one page. Returns list of dicts."""
    truncated = content[:MAX_CHARS]
    prompt = PROMPT_TEMPLATE.format(
        n=QA_PER_PAGE,
        url=page_info["url"],
        title=page_info["title"],
        content=truncated,
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            pairs = data.get("pairs", [])
            if pairs and all(
                isinstance(p, dict) and "question" in p and "answer" in p
                for p in pairs
            ):
                return pairs
            print(f"  ! Malformed JSON structure, retrying...")
        except json.JSONDecodeError as e:
            print(f"  ! JSON parse error (attempt {attempt+1}): {e}")
        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  ! Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  ! Error (attempt {attempt+1}): {e}")
            time.sleep(5)

    print("  x Failed after 3 attempts, skipping this page")
    return []


def save_csv(pairs, path):
    """Write all Q/A pairs to a CSV file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "source_page", "source_title"],
        )
        writer.writeheader()
        writer.writerows(pairs)


def main():
    if not INDEX_FILE.exists():
        print(f"ERROR: {INDEX_FILE} not found. Run scraper.py first.")
        return

    with open(INDEX_FILE) as f:
        index = json.load(f)

    print(f"Loaded {len(index)} pages from index")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env")
        return
    client = Groq(api_key=api_key)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_pairs = []

    for i, page in enumerate(index, 1):
        print(f"\n[{i}/{len(index)}] {page['filename']}")

        filepath = RAW_DIR / page["filename"]
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # Strip the URL/Title header we wrote in scraper.py
        if "---" in content:
            content = content.split("---", 1)[1].strip()

        pairs = generate_qa_for_page(client, page, content)

        for p in pairs:
            all_pairs.append({
                "question": p["question"],
                "answer": p["answer"],
                "source_page": page["url"],
                "source_title": page["title"],
            })

        print(f"  v Got {len(pairs)} pairs (running total: {len(all_pairs)})")

        # Save incrementally so partial progress is preserved if it crashes
        if i % 5 == 0 or i == len(index):
            save_csv(all_pairs, OUTPUT_CSV)
            print(f"  > Saved progress to {OUTPUT_CSV}")

        time.sleep(REQUEST_DELAY)

    save_csv(all_pairs, OUTPUT_CSV)

    print("\n" + "=" * 60)
    print(f"DONE. Generated {len(all_pairs)} Q/A pairs from {len(index)} pages")
    print(f"Saved to: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()