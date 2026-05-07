from __future__ import annotations
"""
Token-aware text chunking with overlap.
"""

import tiktoken
import config


def chunk_documents(
    documents: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    model: str = "cl100k_base",
) -> list[dict]:
    """
    Split documents into overlapping token-based chunks.

    Args:
        documents: list of {"doc_id": str, "text": str}
        chunk_size: max tokens per chunk
        overlap: token overlap between consecutive chunks
    Returns:
        list of {"chunk_id": str, "doc_id": str, "text": str}
    """
    enc = tiktoken.get_encoding(model)
    chunks = []
    chunk_counter = 0

    for doc in documents:
        tokens = enc.encode(doc["text"])
        if len(tokens) <= chunk_size:
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_id": doc["doc_id"],
                "text": doc["text"],
            })
            chunk_counter += 1
            continue

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "doc_id": doc["doc_id"],
                "text": chunk_text,
            })
            chunk_counter += 1
            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample_docs = [
        {"doc_id": "test", "text": "Albert Einstein was born in Ulm, Germany. " * 50}
    ]
    result = chunk_documents(sample_docs, chunk_size=50, overlap=10)
    print(f"Created {len(result)} chunks from {len(sample_docs)} documents")
    for c in result[:3]:
        print(f"  {c['chunk_id']}: {c['text'][:80]}...")
