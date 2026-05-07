from __future__ import annotations
"""
Download and parse HotpotQA and MuSiQue benchmark datasets.
Returns standardized records: {question, answer, supporting_facts, context_paragraphs}.
"""

import json
from typing import Optional
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import config


def load_hotpotqa(split: str = "validation", n_samples: Optional[int] = None) -> list:
    """Load HotpotQA distractor setting from HuggingFace."""
    cache_path = config.DATA_RAW / f"hotpotqa_{split}_{n_samples}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    records = []
    for i, row in enumerate(tqdm(ds, desc="Loading HotpotQA")):
        if n_samples and i >= n_samples:
            break
        paragraphs = []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            paragraphs.append({
                "title": title,
                "text": " ".join(sentences),
            })
        records.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "supporting_facts_titles": list(set(row["supporting_facts"]["title"])),
            "context_paragraphs": paragraphs,
        })

    cache_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def load_musique(split: str = "validation", n_samples: Optional[int] = None) -> list:
    """Load MuSiQue dataset from HuggingFace."""
    cache_path = config.DATA_RAW / f"musique_{split}_{n_samples}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    ds = load_dataset("drt/musique", split=split, trust_remote_code=True)
    records = []
    for i, row in enumerate(tqdm(ds, desc="Loading MuSiQue")):
        if n_samples and i >= n_samples:
            break
        paragraphs = []
        if row.get("paragraphs"):
            for p in row["paragraphs"]:
                paragraphs.append({
                    "title": p.get("title", ""),
                    "text": p.get("paragraph_text", ""),
                    "is_supporting": p.get("is_supporting", False),
                })
        records.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row.get("answer", ""),
            "answerable": row.get("answerable", True),
            "context_paragraphs": paragraphs,
        })

    cache_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def get_all_paragraphs(records: list) -> list:
    """Flatten all context paragraphs from a dataset into a single list with doc_ids."""
    paragraphs = []
    seen = set()
    for rec in records:
        for para in rec["context_paragraphs"]:
            key = (para["title"], para["text"][:100])
            if key not in seen:
                seen.add(key)
                paragraphs.append({
                    "doc_id": para["title"],
                    "text": para["text"],
                })
    return paragraphs


if __name__ == "__main__":
    print("Loading HotpotQA...")
    hqa = load_hotpotqa(n_samples=5)
    print(f"  {len(hqa)} records loaded. Sample question: {hqa[0]['question']}")

    print("Loading MuSiQue...")
    muq = load_musique(n_samples=5)
    print(f"  {len(muq)} records loaded. Sample question: {muq[0]['question']}")
