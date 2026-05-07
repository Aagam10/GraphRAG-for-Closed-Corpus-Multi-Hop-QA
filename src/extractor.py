from __future__ import annotations
"""
LLM-based entity/relation triple extraction from text chunks.

This is the most critical module: quality of the knowledge graph depends entirely
on extraction precision and entity normalization.
"""

import json
import re
import time
from pathlib import Path
from difflib import SequenceMatcher
from tqdm import tqdm
import config


client = config.get_llm_client()

# Load prompt template
_EXTRACTION_PROMPT = (config.PROMPTS_DIR / "extraction.txt").read_text(encoding="utf-8")


# ── Entity normalization ─────────────────────────────────────────────

# Common aliases → canonical form
_ALIASES = {
    "us": "united states",
    "usa": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "nyc": "new york city",
    "ny": "new york",
    "dc": "washington, d.c.",
    "la": "los angeles",
}


def normalize_entity(name: str) -> str:
    """Normalize an entity name: lowercase, strip, resolve common aliases."""
    name = name.strip().lower()
    # Remove trailing periods
    name = name.rstrip(".")
    # Collapse whitespace
    name = " ".join(name.split())
    # Resolve known aliases
    return _ALIASES.get(name, name)


def _merge_similar_entities(triples: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Merge entities with high string similarity (e.g., "albert einstein" and "a. einstein").
    Uses a simple greedy approach: build a canonical map, then rewrite all triples.
    """
    all_entities = set()
    for t in triples:
        all_entities.add(t["entity_a"])
        all_entities.add(t["entity_b"])

    entities = sorted(all_entities)
    canonical = {}  # entity → canonical form

    for e in entities:
        if e in canonical:
            continue
        canonical[e] = e
        for other in entities:
            if other in canonical or other == e:
                continue
            # Check if one is a substring of the other, or high similarity
            if e in other or other in e:
                # Keep the longer form as canonical
                canon = e if len(e) >= len(other) else other
                canonical[other] = canon
                canonical[e] = canon
            elif SequenceMatcher(None, e, other).ratio() >= threshold:
                canon = e if len(e) >= len(other) else other
                canonical[other] = canon
                canonical[e] = canon

    # Rewrite triples
    for t in triples:
        t["entity_a"] = canonical.get(t["entity_a"], t["entity_a"])
        t["entity_b"] = canonical.get(t["entity_b"], t["entity_b"])

    return triples


# ── LLM extraction ───────────────────────────────────────────────────

def _parse_llm_response(response_text: str) -> list[dict]:
    """Parse the LLM response into a list of triple dicts, handling common malformations."""
    text = response_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    valid = []
    for item in data:
        if isinstance(item, dict) and all(k in item for k in ("entity_a", "relation", "entity_b")):
            valid.append({
                "entity_a": normalize_entity(str(item["entity_a"])),
                "relation": str(item["relation"]).strip().lower().replace(" ", "_"),
                "entity_b": normalize_entity(str(item["entity_b"])),
            })
    return valid


def extract_triples_from_chunk(
    chunk_text: str,
    chunk_id: str = "",
    model: str = config.EXTRACTION_MODEL,
) -> list[dict]:
    """Extract triples from a single text chunk via LLM."""
    prompt = _EXTRACTION_PROMPT.replace("{text}", chunk_text)

    for attempt in range(config.MAX_EXTRACTION_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content
            triples = _parse_llm_response(raw)
            # Tag with source
            for t in triples:
                t["source_chunk_id"] = chunk_id
            return triples
        except Exception as e:
            err_str = str(e)
            # Parse retry delay from API error if available
            retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
            if retry_match:
                wait = int(float(retry_match.group(1))) + 2
            else:
                wait = 15 * (attempt + 1)  # 15s, 30s, 45s

            if attempt < config.MAX_EXTRACTION_RETRIES - 1:
                print(f"  Extraction error (attempt {attempt+1}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Extraction failed for chunk {chunk_id}: {e}")
                return []


def extract_triples(
    chunks: list[dict],
    model: str = config.EXTRACTION_MODEL,
    cache_path: Path = config.TRIPLES_CACHE,
    batch_size: int = 10,
) -> list[dict]:
    """
    Extract triples from all chunks. Caches results to disk.

    Args:
        chunks: list of {"chunk_id": str, "doc_id": str, "text": str}
    Returns:
        list of {"entity_a", "relation", "entity_b", "source_chunk_id"}
    """
    # Check cache
    if cache_path.exists():
        print(f"Loading cached triples from {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    all_triples = []
    # Rate limiting: Gemini free=4s, Groq free=0, Bedrock paid=0
    if "gemini" in model.lower():
        rate_limit_delay = 4
    else:
        rate_limit_delay = 0

    for i, chunk in enumerate(tqdm(chunks, desc="Extracting triples")):
        triples = extract_triples_from_chunk(
            chunk["text"],
            chunk_id=chunk["chunk_id"],
            model=model,
        )
        all_triples.extend(triples)

        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        # Periodic save every batch_size chunks
        if (i + 1) % batch_size == 0:
            cache_path.write_text(
                json.dumps(all_triples, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # Merge similar entities
    all_triples = _merge_similar_entities(all_triples)

    # Deduplicate identical triples
    seen = set()
    deduped = []
    for t in all_triples:
        key = (t["entity_a"], t["relation"], t["entity_b"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    all_triples = deduped

    # Final save
    cache_path.write_text(
        json.dumps(all_triples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Extracted {len(all_triples)} unique triples from {len(chunks)} chunks")
    return all_triples


if __name__ == "__main__":
    # Test with a small example
    test_chunks = [{
        "chunk_id": "test_0",
        "doc_id": "test",
        "text": "Marie Curie was born in Warsaw, Poland. She moved to Paris and studied at the University of Paris.",
    }]
    triples = extract_triples(test_chunks, cache_path=Path("test_triples.json"))
    for t in triples:
        print(f"  ({t['entity_a']}) --[{t['relation']}]--> ({t['entity_b']})")
