"""
src/retriever.py

Hybrid retrieval over two Chroma collections:

  1. First, search vizcom_qa for the user's question. If the top match is
     close enough (cosine distance < QA_MATCH_THRESHOLD), return that
     pre-written answer directly. Fast + grounded in our curated Q/A set.

  2. Otherwise, fall back: search vizcom_docs for relevant chunks and
     return them as context for the LLM to synthesize an answer.

This is the "Not X -> fallback" logic from the lecture whiteboard.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


CHROMA_DIR = Path("chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Tuning knobs (you may tweak these later during testing)
QA_MATCH_THRESHOLD = 0.40   # cosine distance; lower = more similar
                            #   0.00 = identical, 1.00 = unrelated
                            #   < 0.40 means "very confident this is the same question"
TOP_K_DOCS = 4              # number of chunks to retrieve in vector fallback


class HybridRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.qa_collection = self.client.get_collection("vizcom_qa")
        self.docs_collection = self.client.get_collection("vizcom_docs")

    def retrieve(self, query: str) -> dict:
        """
        Run hybrid retrieval. Returns a dict with:
          mode             : 'qa' (direct Q/A hit) or 'vector' (LLM needed)
          answer           : pre-written answer, only in 'qa' mode
          matched_question : Q/A question that matched, only in 'qa' mode
          context          : list of {text, url, title}, only in 'vector' mode
          sources          : list of {url, title}
          confidence       : distance score (lower = better)
        """
        # 1. Embed the query
        query_embedding = self.embedder.encode([query]).tolist()

        # 2. Stage 1 - search Q/A collection
        qa_results = self.qa_collection.query(
            query_embeddings=query_embedding,
            n_results=1,
        )

        if qa_results["distances"] and qa_results["distances"][0]:
            top_distance = qa_results["distances"][0][0]
            top_meta = qa_results["metadatas"][0][0]

            if top_distance < QA_MATCH_THRESHOLD:
                # High-confidence direct Q/A match - return immediately
                return {
                    "mode": "qa",
                    "answer": top_meta["answer"],
                    "matched_question": top_meta["question"],
                    "context": [],
                    "sources": [{
                        "url": top_meta["source_page"],
                        "title": top_meta["source_title"],
                    }],
                    "confidence": top_distance,
                }

        # 3. Stage 2 - fall back to vector search over the docs
        doc_results = self.docs_collection.query(
            query_embeddings=query_embedding,
            n_results=TOP_K_DOCS,
        )

        context = []
        sources_seen = set()
        sources = []
        for i in range(len(doc_results["documents"][0])):
            text = doc_results["documents"][0][i]
            meta = doc_results["metadatas"][0][i]
            context.append({
                "text": text,
                "url": meta["url"],
                "title": meta["title"],
            })
            if meta["url"] not in sources_seen:
                sources_seen.add(meta["url"])
                sources.append({
                    "url": meta["url"],
                    "title": meta["title"],
                })

        return {
            "mode": "vector",
            "answer": None,
            "matched_question": None,
            "context": context,
            "sources": sources,
            "confidence": (
                doc_results["distances"][0][0]
                if doc_results["distances"][0] else None
            ),
        }


# ---- Quick test when run directly ----
if __name__ == "__main__":
    r = HybridRetriever()

    test_queries = [
        "What is Vizcom?",                # should hit Q/A directly
        "How do I get started with Vizcom?",  # should hit Q/A directly
        "Can I export my designs to STL?",    # might fall back to vector
        "Tell me about color variations",     # niche - likely vector
        "What's the meaning of life?",        # off-topic - vector w/ weak match
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        result = r.retrieve(q)
        print(f"MODE:  {result['mode']}  "
              f"(confidence dist: {result['confidence']:.3f})")
        if result["mode"] == "qa":
            print(f"MATCHED Q: {result['matched_question']}")
            print(f"ANSWER:    {result['answer']}")
        else:
            print(f"RETRIEVED {len(result['context'])} chunks")
            print(f"TOP CHUNK: {result['context'][0]['text'][:200]}...")
        print(f"SOURCES:   {[s['title'][:50] for s in result['sources']]}")