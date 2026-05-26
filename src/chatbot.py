"""
src/chatbot.py

The Vizcom chatbot. Uses HybridRetriever to find relevant info, then:

  - If a Q/A pair matched directly, return that curated answer.
  - If vector search found decent matches, call Groq to synthesize an
    answer from those chunks.
  - If even vector search has weak confidence, politely say "I don't know"
    instead of hallucinating.

Maintains short conversation history for multi-turn follow-ups.
"""

import os
import sys
from pathlib import Path

# Make sibling modules importable when running this file directly
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from groq import Groq

from retriever import HybridRetriever  # noqa: E402


load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 4            # how many past Q/A pairs to include for context
LOW_CONFIDENCE_THRESHOLD = 1.0   # vector distance > this = decline to answer


SYSTEM_PROMPT = """You are a friendly, professional assistant for Vizcom -
an AI design platform for product designers and creative teams.

Your job is to answer questions about Vizcom using ONLY the context provided
to you. Do not invent features, pricing, capabilities, or facts not in the
context.

Style:
- Be concise. 2-4 sentences unless the user asks for more detail.
- Use a warm but professional tone.
- If the context does not contain the answer, say so clearly and suggest
  the user visit vizcom.com or contact the Vizcom team."""


class VizcomChatbot:
    def __init__(self):
        self.retriever = HybridRetriever()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        self.client = Groq(api_key=api_key)
        self.history = []  # list of (user_q, assistant_a) tuples

    def reset_history(self):
        self.history = []

    def _needs_rewrite(self, query: str) -> bool:
        """Detect if a query likely depends on conversation context."""
        if not self.history:
            return False
        q_lower = " " + query.lower() + " "
        # Pronouns/references that usually need resolving
        ambiguous = [
            " it ", " that ", " this ", " these ", " those ",
            " them ", " they ", " its ", " their ",
        ]
        return any(token in q_lower for token in ambiguous)

    def _rewrite_query(self, query: str) -> str:
        """Use the LLM to turn a context-dependent query into a standalone one."""
        if not self._needs_rewrite(query):
            return query

        recent_history = self.history[-3:]  # last 3 turns is plenty
        history_text = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in recent_history
        )

        rewrite_prompt = (
            "Given the conversation history below, rewrite the user's follow-up "
            "question into a standalone question that can be understood without "
            "the history. Resolve pronouns (it, that, this) to the specific thing "
            "they refer to. If the question is already standalone, return it "
            "unchanged.\n\n"
            "Return ONLY the rewritten question - no explanation, no quotes.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {query}\n\n"
            "Standalone question:"
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        print(f"  [query rewrite] '{query}' -> '{rewritten}'")
        return rewritten

    def ask(self, query: str) -> dict:
        """Answer a user query with optional history-aware query rewriting."""
        # Step 0: rewrite the query if it depends on prior context
        effective_query = self._rewrite_query(query)

        # Step 1: retrieve using the (possibly rewritten) query
        result = self.retriever.retrieve(effective_query)

        # Case 1: high-confidence Q/A match
        if result["mode"] == "qa":
            answer = result["answer"]
            self.history.append((query, answer))
            return {
                "answer": answer,
                "mode": "qa",
                "sources": result["sources"],
                "confidence": result["confidence"],
                "matched_question": result["matched_question"],
                "rewritten_query": effective_query if effective_query != query else None,
            }

        # Case 2: vector match too weak - decline gracefully
        if (
            result["confidence"] is not None
            and result["confidence"] > LOW_CONFIDENCE_THRESHOLD
        ):
            answer = (
                "I don't have specific information about that in my Vizcom "
                "knowledge base. You can check vizcom.com or reach out to "
                "the Vizcom team for more details."
            )
            self.history.append((query, answer))
            return {
                "answer": answer,
                "mode": "unknown",
                "sources": [],
                "confidence": result["confidence"],
                "matched_question": None,
                "rewritten_query": effective_query if effective_query != query else None,
            }

        # Case 3: vector match -> generate via Groq
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['title']}]\n{c['text']}"
            for c in result["context"]
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for past_q, past_a in self.history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": past_q})
            messages.append({"role": "assistant", "content": past_a})

        user_message = (
            "Use the following context from Vizcom's website to answer the question.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {effective_query}\n"
        )
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )

        answer = response.choices[0].message.content.strip()
        self.history.append((query, answer))

        return {
            "answer": answer,
            "mode": "vector",
            "sources": result["sources"],
            "confidence": result["confidence"],
            "matched_question": None,
            "rewritten_query": effective_query if effective_query != query else None,
        }

        # Build messages including short conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for past_q, past_a in self.history[-MAX_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": past_q})
            messages.append({"role": "assistant", "content": past_a})

        user_message = f"""Use the following context from Vizcom's website to answer the question.

CONTEXT:
{context_text}

QUESTION: {query}
"""
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )

        answer = response.choices[0].message.content.strip()
        self.history.append((query, answer))

        return {
            "answer": answer,
            "mode": "vector",
            "sources": result["sources"],
            "confidence": result["confidence"],
            "matched_question": None,
        }


# ---- Quick test ----
if __name__ == "__main__":
    bot = VizcomChatbot()

    test_queries = [
        "What is Vizcom?",
        "How does it help with car design?",   # follow-up using 'it'
        "Can I export to STL files?",
        "What's the meaning of life?",         # off-topic - should say IDK
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"USER: {q}")
        result = bot.ask(q)
        print(f"MODE: {result['mode']} (confidence: {result['confidence']:.3f})")
        if result.get("matched_question"):
            print(f"MATCHED Q: {result['matched_question']}")
        print(f"\nANSWER: {result['answer']}")
        if result['sources']:
            print(f"\nSOURCES:")
            for s in result['sources']:
                print(f"  - {s['title'][:60]}: {s['url']}")
