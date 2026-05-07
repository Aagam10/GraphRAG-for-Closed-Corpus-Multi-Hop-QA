from __future__ import annotations
"""
Vanilla RAG baseline: FAISS vector search + BM25 retrieval for comparison with GraphRAG.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI
import config

client = config.get_llm_client()

_VANILLA_PROMPT = """Answer the question using ONLY the context provided below.

## Context
{context}

## Question
{question}

## Answer
Your LAST line MUST be exactly: "Final answer: <your answer>"
The final answer must be SHORT — a word, name, yes/no, or brief phrase.
If the context is insufficient, write: "Final answer: I don't have enough information"
"""


class VanillaRAG:
    """FAISS + BM25 hybrid retrieval baseline."""

    def __init__(self, chunks: list[dict], embedding_model: str = config.EMBEDDING_MODEL):
        """
        Args:
            chunks: list of {"chunk_id": str, "doc_id": str, "text": str}
        """
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.doc_ids = [c["doc_id"] for c in chunks]

        # Build BM25 index
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Build FAISS index
        print("Building FAISS index...")
        self.embed_model = SentenceTransformer(embedding_model)
        self.embeddings = self.embed_model.encode(self.texts, show_progress_bar=True)
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def retrieve(self, query: str, top_k: int = config.VANILLA_TOP_K) -> list[dict]:
        """Hybrid retrieval: combine BM25 and FAISS scores."""
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # FAISS (cosine similarity)
        q_emb = self.embed_model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        faiss_scores = np.dot(self.embeddings, q_emb.T).flatten()

        # Combine: equal weight
        combined = 0.5 * bm25_scores + 0.5 * faiss_scores
        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "doc_id": self.chunks[idx]["doc_id"],
                "text": self.chunks[idx]["text"],
                "score": float(combined[idx]),
            })
        return results

    def answer(self, question: str, top_k: int = config.VANILLA_TOP_K, model: str = config.GENERATION_MODEL) -> dict:
        """Retrieve chunks and generate an answer."""
        retrieved = self.retrieve(question, top_k)
        context = "\n\n".join(
            f"[Document: {r['doc_id']}]\n{r['text']}" for r in retrieved
        )

        prompt = _VANILLA_PROMPT.replace("{question}", question).replace("{context}", context)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            answer = response.choices[0].message.content.strip()
            # Extract short final answer
            for marker in ["final answer:", "the answer is:", "answer:"]:
                idx = answer.lower().rfind(marker)
                if idx != -1:
                    extracted = answer[idx + len(marker):].strip()
                    first_line = extracted.split("\n")[0].strip().rstrip(".")
                    if first_line:
                        answer = first_line
                        break
        except Exception as e:
            print(f"Vanilla generation error: {e}")
            answer = ""

        return {
            "answer": answer,
            "retrieved_titles": list(set(r["doc_id"] for r in retrieved)),
        }


if __name__ == "__main__":
    sample_chunks = [
        {"chunk_id": "0", "doc_id": "Einstein", "text": "Albert Einstein was born in Ulm, Germany in 1879."},
        {"chunk_id": "1", "doc_id": "Ulm", "text": "Ulm is a city in the German state of Baden-Württemberg."},
        {"chunk_id": "2", "doc_id": "Relativity", "text": "The theory of relativity was developed by Albert Einstein."},
    ]
    rag = VanillaRAG(sample_chunks)
    result = rag.answer("Where was the developer of relativity born?")
    print(f"Answer: {result['answer']}")
    print(f"Retrieved: {result['retrieved_titles']}")
