from __future__ import annotations
"""
Multi-hop retrieval: entity linking + k-hop subgraph extraction + relevance pruning.

This is the core retrieval mechanism that differentiates GraphRAG from vanilla RAG.
The key challenges are:
  1. Entity linking: matching query mentions to graph nodes (fuzzy + embedding fallback)
  2. Subgraph pruning: the k-hop neighborhood can explode; we must rank and prune paths
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from openai import OpenAI
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
import config

client = config.get_llm_client()
_QUERY_PROMPT = (config.PROMPTS_DIR / "query_entities.txt").read_text(encoding="utf-8")

# Lazy-loaded embedding model for entity linking fallback
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


# ── Step 1: Extract query entities via LLM ───────────────────────────

def extract_query_entities(question: str, model: str = config.EXTRACTION_MODEL) -> list[str]:
    """Use an LLM to extract key entities from a question."""
    prompt = _QUERY_PROMPT.replace("{question}", question)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        entities = json.loads(raw)
        if isinstance(entities, list):
            return [e.strip().lower() for e in entities if isinstance(e, str)]
    except Exception as e:
        print(f"  Query entity extraction error: {e}")

    # Fallback: naive NER — extract capitalized phrases
    words = question.split()
    entities = []
    current = []
    for w in words:
        if w[0].isupper() and w.lower() not in {"what", "who", "where", "when", "which", "how", "is", "was", "the", "a", "an", "in", "of", "at", "for"}:
            current.append(w.lower())
        else:
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))
    return entities


# ── Step 2: Entity linking — map query entities to graph nodes ───────

def link_entities(
    query_entities: list[str],
    G: nx.DiGraph,
    threshold: float = config.FUZZY_MATCH_THRESHOLD,
) -> list[str]:
    """
    Map query entity strings to actual graph node IDs using a multi-strategy approach:
      1. Exact match
      2. Substring containment (either direction)
      3. Token-level overlap (new: handles "ed wood" → "ed wood sr")
      4. Fuzzy string similarity (lowered threshold)
      5. Embedding similarity fallback
    """
    graph_nodes = list(G.nodes())
    if not graph_nodes:
        return []

    matched = set()

    for qe in query_entities:
        qe_lower = qe.lower().strip()
        qe_tokens = set(qe_lower.split())

        # Strategy 1: Exact match
        if qe_lower in G:
            matched.add(qe_lower)
            continue

        # Strategy 2: Substring containment (either direction)
        substring_matches = []
        for node in graph_nodes:
            if qe_lower in node or node in qe_lower:
                substring_matches.append(node)
        if substring_matches:
            best = min(substring_matches, key=lambda n: abs(len(n) - len(qe_lower)))
            matched.add(best)
            continue

        # Strategy 3: Token-level overlap (new)
        # e.g. "ed wood" tokens {ed, wood} match "ed wood sr" tokens {ed, wood, sr}
        token_matches = []
        for node in graph_nodes:
            node_tokens = set(node.split())
            overlap = len(qe_tokens & node_tokens)
            if overlap > 0 and overlap >= len(qe_tokens) * 0.6:
                token_matches.append((node, overlap / max(len(qe_tokens), len(node_tokens))))
        if token_matches:
            token_matches.sort(key=lambda x: x[1], reverse=True)
            matched.add(token_matches[0][0])
            continue

        # Strategy 4: Fuzzy matching (lowered threshold from 0.75 to 0.55)
        best_score = 0
        best_node = None
        for node in graph_nodes:
            score = SequenceMatcher(None, qe_lower, node).ratio()
            if score > best_score:
                best_score = score
                best_node = node
        if best_score >= 0.55:  # lowered from config threshold
            matched.add(best_node)
            continue

        # Strategy 5: Embedding similarity fallback
        try:
            model = _get_embed_model()
            qe_emb = model.encode([qe_lower])
            node_embs = model.encode(graph_nodes)
            sims = np.dot(node_embs, qe_emb.T).flatten()
            top_idx = np.argmax(sims)
            if sims[top_idx] >= 0.45:  # lowered embedding threshold
                matched.add(graph_nodes[top_idx])
        except Exception:
            pass

    return list(matched)


def keyword_fallback_nodes(question: str, G: nx.DiGraph, top_k: int = 5) -> list[str]:
    """
    Last-resort fallback: scan ALL graph nodes for question keyword overlap.
    Used when entity linking returns zero seed nodes.
    """
    q_tokens = set(question.lower().split()) - {
        "what", "who", "where", "when", "which", "how", "is", "was", "were",
        "the", "a", "an", "in", "of", "at", "for", "and", "or", "to", "did",
        "does", "do", "by", "with", "that", "this", "are", "be", "been",
        "same", "different", "located", "held", "based", "portrayed",
    }
    scored = []
    for node in G.nodes():
        node_tokens = set(node.split())
        overlap = len(q_tokens & node_tokens)
        if overlap > 0:
            scored.append((node, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:top_k]]


# ── Step 3: k-hop subgraph extraction ────────────────────────────────

def extract_subgraph(
    G: nx.DiGraph,
    seed_nodes: list[str],
    k_hops: int = config.K_HOPS,
) -> nx.DiGraph:
    """
    BFS traversal from seed nodes up to k hops in both directions.
    Returns the induced subgraph.
    """
    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(k_hops):
        next_frontier = set()
        for node in frontier:
            # Outgoing neighbors
            if node in G:
                for neighbor in G.successors(node):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            # Incoming neighbors (traverse edges in both directions)
            if node in G:
                for neighbor in G.predecessors(node):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return G.subgraph(visited).copy()


# ── Step 4: Subgraph pruning by relevance ────────────────────────────

def prune_subgraph(
    subgraph: nx.DiGraph,
    query_entities: list[str],
    seed_nodes: list[str],
    top_n: int = config.TOP_N_PATHS,
) -> nx.DiGraph:
    """
    Rank edges in the subgraph by relevance and keep only top_n.
    Relevance scoring:
      - Edges connecting seed nodes get highest score
      - Edges with nodes that share tokens with query entities get a boost
      - Edge weight (frequency) serves as a tiebreaker
    """
    if subgraph.number_of_edges() <= top_n:
        return subgraph

    query_tokens = set()
    for qe in query_entities:
        query_tokens.update(qe.lower().split())

    scored_edges = []
    seed_set = set(seed_nodes)

    for u, v, data in subgraph.edges(data=True):
        score = 0.0
        # Proximity to seed nodes
        if u in seed_set:
            score += 3.0
        if v in seed_set:
            score += 3.0

        # Token overlap with query
        u_tokens = set(u.lower().split())
        v_tokens = set(v.lower().split())
        overlap = len((u_tokens | v_tokens) & query_tokens)
        score += overlap * 1.5

        # Edge weight (frequency)
        score += data.get("weight", 1) * 0.5

        # Degree centrality (prefer hub nodes — they're often important entities)
        score += (subgraph.degree(u) + subgraph.degree(v)) * 0.1

        scored_edges.append((u, v, score))

    # Keep top_n edges
    scored_edges.sort(key=lambda x: x[2], reverse=True)
    keep_edges = scored_edges[:top_n]

    # Build pruned subgraph
    pruned = nx.DiGraph()
    for u, v, _ in keep_edges:
        pruned.add_edge(u, v, **subgraph.edges[u, v])
        pruned.nodes[u].update(subgraph.nodes[u])
        pruned.nodes[v].update(subgraph.nodes[v])

    return pruned


# ── Step 5: Linearize subgraph into text ─────────────────────────────

def linearize_subgraph(subgraph: nx.DiGraph) -> str:
    """Convert a subgraph into readable text facts for the LLM prompt."""
    lines = []
    for i, (u, v, data) in enumerate(subgraph.edges(data=True), 1):
        relations = data.get("relations", ["related_to"])
        rel_str = ", ".join(set(relations))
        lines.append(f"Fact {i}: [{u}] --({rel_str})--> [{v}]")
    return "\n".join(lines)


# ── Full retrieval pipeline ──────────────────────────────────────────

def retrieve(
    question: str,
    G: nx.DiGraph,
    k_hops: int = config.K_HOPS,
    top_n: int = config.TOP_N_PATHS,
) -> dict:
    """
    Full retrieval pipeline: question → entities → link → traverse → prune → linearize.

    Returns:
        {
            "query_entities": [...],
            "seed_nodes": [...],
            "subgraph": nx.DiGraph,
            "linearized": str,
            "retrieved_titles": [...],  # doc_ids from source chunks for eval
        }
    """
    query_entities = extract_query_entities(question)
    seed_nodes = link_entities(query_entities, G)

    # Fallback: keyword scan over all nodes if entity linking found nothing
    if not seed_nodes:
        seed_nodes = keyword_fallback_nodes(question, G)

    if not seed_nodes:
        return {
            "query_entities": query_entities,
            "seed_nodes": [],
            "subgraph": nx.DiGraph(),
            "linearized": "No relevant facts found in the knowledge graph.",
            "retrieved_titles": [],
        }

    subgraph = extract_subgraph(G, seed_nodes, k_hops)
    pruned = prune_subgraph(subgraph, query_entities, seed_nodes, top_n)
    linearized = linearize_subgraph(pruned)

    # Collect source doc_ids for retrieval evaluation
    retrieved_titles = set()
    for n in pruned.nodes():
        for src in pruned.nodes[n].get("source_chunks", []):
            retrieved_titles.add(src)

    return {
        "query_entities": query_entities,
        "seed_nodes": seed_nodes,
        "subgraph": pruned,
        "linearized": linearized,
        "retrieved_titles": list(retrieved_titles),
    }


if __name__ == "__main__":
    from graph_builder import build_graph

    test_triples = [
        {"entity_a": "albert einstein", "relation": "born_in", "entity_b": "ulm", "source_chunk_id": "c1"},
        {"entity_a": "ulm", "relation": "located_in", "entity_b": "germany", "source_chunk_id": "c1"},
        {"entity_a": "albert einstein", "relation": "developed", "entity_b": "theory of relativity", "source_chunk_id": "c2"},
        {"entity_a": "theory of relativity", "relation": "published_in", "entity_b": "1905", "source_chunk_id": "c2"},
        {"entity_a": "germany", "relation": "is_country_in", "entity_b": "europe", "source_chunk_id": "c1"},
    ]
    G = build_graph(test_triples)
    result = retrieve("What country is the birthplace of the developer of the theory of relativity?", G)
    print(f"Query entities: {result['query_entities']}")
    print(f"Seed nodes: {result['seed_nodes']}")
    print(f"Subgraph: {result['subgraph'].number_of_nodes()} nodes, {result['subgraph'].number_of_edges()} edges")
    print(f"Context:\n{result['linearized']}")
