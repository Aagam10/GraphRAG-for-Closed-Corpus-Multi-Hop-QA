from __future__ import annotations
"""
Build and persist a NetworkX directed knowledge graph from extracted triples.
"""

import json
import pickle
from pathlib import Path
from collections import Counter
import networkx as nx
import config


def build_graph(triples: list[dict]) -> nx.DiGraph:
    """
    Build a directed graph from triples.

    Args:
        triples: list of {"entity_a": str, "relation": str, "entity_b": str, "source_chunk_id": str}
    Returns:
        networkx.DiGraph
    """
    G = nx.DiGraph()

    for t in triples:
        a = t["entity_a"]
        b = t["entity_b"]
        rel = t["relation"]
        src = t.get("source_chunk_id", "")

        # Add/update node A
        if G.has_node(a):
            G.nodes[a]["mentions"] += 1
            G.nodes[a]["source_chunks"].add(src)
        else:
            G.add_node(a, mentions=1, source_chunks={src})

        # Add/update node B
        if G.has_node(b):
            G.nodes[b]["mentions"] += 1
            G.nodes[b]["source_chunks"].add(src)
        else:
            G.add_node(b, mentions=1, source_chunks={src})

        # Add/update edge
        if G.has_edge(a, b):
            G.edges[a, b]["relations"].append(rel)
            G.edges[a, b]["weight"] += 1
            G.edges[a, b]["source_chunks"].add(src)
        else:
            G.add_edge(a, b, relations=[rel], weight=1, source_chunks={src})

    return G


def save_graph(G: nx.DiGraph, path: Path = config.GRAPH_PATH) -> None:
    """Serialize graph to disk."""
    # Convert sets to lists for pickling
    for n in G.nodes:
        G.nodes[n]["source_chunks"] = list(G.nodes[n].get("source_chunks", set()))
    for u, v in G.edges:
        G.edges[u, v]["source_chunks"] = list(G.edges[u, v].get("source_chunks", set()))

    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path: Path = config.GRAPH_PATH) -> nx.DiGraph:
    """Load graph from disk."""
    with open(path, "rb") as f:
        G = pickle.load(f)
    # Restore sets
    for n in G.nodes:
        G.nodes[n]["source_chunks"] = set(G.nodes[n].get("source_chunks", []))
    for u, v in G.edges:
        G.edges[u, v]["source_chunks"] = set(G.edges[u, v].get("source_chunks", []))
    return G


def get_graph_stats(G: nx.DiGraph) -> dict:
    """Compute summary statistics of the graph."""
    degrees = [d for _, d in G.degree()]
    relation_counts = Counter()
    for _, _, data in G.edges(data=True):
        for r in data.get("relations", []):
            relation_counts[r] += 1

    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_connected_components": len(components),
        "largest_component_size": max(len(c) for c in components) if components else 0,
        "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "top_relations": relation_counts.most_common(10),
    }


if __name__ == "__main__":
    sample_triples = [
        {"entity_a": "albert einstein", "relation": "born_in", "entity_b": "ulm", "source_chunk_id": "c1"},
        {"entity_a": "ulm", "relation": "located_in", "entity_b": "germany", "source_chunk_id": "c1"},
        {"entity_a": "albert einstein", "relation": "developed", "entity_b": "theory of relativity", "source_chunk_id": "c2"},
    ]
    G = build_graph(sample_triples)
    stats = get_graph_stats(G)
    for k, v in stats.items():
        print(f"  {k}: {v}")
