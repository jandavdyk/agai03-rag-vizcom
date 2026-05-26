# Vizcom RAG Chatbot

A document-grounded chatbot built on [vizcom.com](https://vizcom.com) content using
synthetic Q&A generation and hybrid retrieval over a Chroma vector store.

Built as the final assignment for the AGAI-03 cohort by Janda van Dyk.

---

## What it does

The chatbot answers natural-language questions about Vizcom — products, pricing,
solutions, careers, security, design workflows — using a two-stage hybrid
retrieval architecture:

1. **Q&A semantic search.** The user's question is matched against a curated
   dataset of 320 synthetic Q&A pairs generated from the scraped content. If
   the top match is sufficiently close (cosine distance < 0.40), the curated
   answer is returned directly.
2. **Vector fallback.** Otherwise the question is matched against chunks of
   the raw scraped pages, and Llama 3.3 70B generates an answer using the
   retrieved context.
3. **Graceful decline.** If even vector search produces a weak match
   (distance > 1.00), the chatbot politely declines rather than hallucinating.

A small LLM-based query rewriter resolves pronouns ("it", "that") against
recent conversation history before retrieval, so multi-turn conversations
work correctly.

---

## Tech stack

| Component             | Choice                                        |
| --------------------- | --------------------------------------------- |
| Language              | Python 3.14                                   |
| Web scraping          | requests + BeautifulSoup4 + lxml              |
| LLM provider          | [Groq](https://groq.com) (free tier)          |
| LLM model             | Llama 3.3 70B Versatile                       |
| Embeddings            | sentence-transformers / all-MiniLM-L6-v2      |
| Vector database       | Chroma (persistent local)                     |
| Chunking              | LangChain RecursiveCharacterTextSplitter      |
| UI                    | Streamlit                                     |
| Secrets               | python-dotenv                                 |

---

## Pipeline overview
The system runs in two phases — offline data preparation and online query handling.

### Offline (one-time, run from terminal)

```mermaid
flowchart TD
    A[vizcom.com sitemap] --> B[scraper.py]
    B --> C["data/raw/*.txt<br/>40 pages"]
    C --> D[qa_generator.py<br/>Groq LLM]
    D --> E["data/qa_dataset.csv<br/>320 Q&A pairs"]
    C --> F[vector_store.py]
    E --> F
    F --> G["chroma_db/vizcom_docs<br/>244 chunks"]
    F --> H["chroma_db/vizcom_qa<br/>320 Q&A pairs"]
```

### Online (each user query)

```mermaid
flowchart TD
    A[User query] --> B[chatbot.py]
    B --> C{Contains pronoun?<br/>+ history?}
    C -->|Yes| D[Query rewriter<br/>Groq LLM]
    C -->|No| E[retriever.py]
    D --> E
    E --> F{Search vizcom_qa}
    F -->|distance &lt; 0.40| G[Return curated answer<br/>Mode: Direct Q&A match]
    F -->|distance ≥ 0.40| H{Search vizcom_docs}
    H -->|distance &lt; 1.00| I[Groq generates<br/>from context<br/>Mode: Vector + LLM]
    H -->|distance ≥ 1.00| J["I don't have that info<br/>Mode: Out of knowledge base"]
    G --> K[Streamlit UI<br/>with citations]
    I --> K
    J --> K
```