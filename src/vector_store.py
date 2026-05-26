"""
src/vector_store.py

Builds two persistent Chroma collections:
  1. vizcom_qa   - the 200 Q/A pairs, indexed by question embedding
  2. vizcom_docs - chunks of the raw scraped pages, for vector fallback

Embeddings are computed with sentence-transformers/all-MiniLM-L6-v2
(small, fast, free, runs locally on CPU).

Run from project root:
    python src/vector_store.py
"""

import csv
import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ---- Configuration ----
RAW_DIR = Path("data/raw")
INDEX_FILE = RAW_DIR / "_index.json"
QA_CSV = Path("data/qa_dataset.csv")
CHROMA_DIR = Path("chroma_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents():
    """Load all raw pages with metadata, stripping the URL/Title header."""
    with open(INDEX_FILE) as f:
        index = json.load(f)

    docs = []
    for page in index:
        filepath = RAW_DIR / page["filename"]
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # Remove the URL/Title header we wrote in scraper.py
        if "---" in content:
            content = content.split("---", 1)[1].strip()

        docs.append({
            "url": page["url"],
            "title": page["title"],
            "filename": page["filename"],
            "text": content,
        })

    return docs


def chunk_documents(docs):
    """Split documents into chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "url": doc["url"],
                "title": doc["title"],
                "filename": doc["filename"],
                "chunk_index": i,
            })

    return all_chunks


def load_qa_pairs():
    """Load the Q/A CSV as a list of dicts."""
    with open(QA_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("=" * 60)
    print("Building Chroma vector store")
    print("=" * 60)

    # 1. Load the embedding model (downloads ~80MB on first run)
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("  Model loaded.")

    # 2. Set up a persistent Chroma client
    print(f"\nInitializing Chroma at: {CHROMA_DIR}")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset collections if they exist (so re-running is idempotent)
    for name in ["vizcom_docs", "vizcom_qa"]:
        try:
            client.delete_collection(name)
            print(f"  Deleted existing collection: {name}")
        except Exception:
            pass

    # 3. Build vizcom_docs (chunks of raw text)
    print("\n--- Building vizcom_docs collection ---")
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks from {len(docs)} pages")

    print("  Embedding chunks (this takes ~30 seconds)...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = embedder.encode(
        chunk_texts, show_progress_bar=True
    ).tolist()

    docs_collection = client.create_collection(name="vizcom_docs")
    docs_collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=[
            {
                "url": c["url"],
                "title": c["title"],
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    print(f"  Added {docs_collection.count()} chunks.")

    # 4. Build vizcom_qa (Q/A pairs, indexed by question)
    print("\n--- Building vizcom_qa collection ---")
    pairs = load_qa_pairs()
    print(f"  Loaded {len(pairs)} Q/A pairs from CSV")

    print("  Embedding questions...")
    questions = [p["question"] for p in pairs]
    question_embeddings = embedder.encode(
        questions, show_progress_bar=True
    ).tolist()

    qa_collection = client.create_collection(name="vizcom_qa")
    qa_collection.add(
        ids=[f"qa_{i}" for i in range(len(pairs))],
        documents=questions,
        embeddings=question_embeddings,
        metadatas=[
            {
                "question": p["question"],
                "answer": p["answer"],
                "source_page": p["source_page"],
                "source_title": p["source_title"],
            }
            for p in pairs
        ],
    )
    print(f"  Added {qa_collection.count()} Q/A pairs.")

    # 5. Summary
    print("\n" + "=" * 60)
    print("DONE.")
    print(f"  vizcom_docs:  {docs_collection.count()} chunks")
    print(f"  vizcom_qa:    {qa_collection.count()} Q/A pairs")
    print(f"  Persisted to: {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()