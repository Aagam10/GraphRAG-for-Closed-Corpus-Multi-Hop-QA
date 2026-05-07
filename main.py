"""
CLI entry point for the GraphRAG pipeline.

Usage:
    python main.py build-graph [--dataset hotpotqa|musique] [--samples N]
    python main.py query "Your question here"
    python main.py evaluate [--dataset hotpotqa|musique] [--samples N]
    python main.py stats
"""

import sys
import json
import argparse

sys.path.insert(0, ".")
import config
from src.data_loader import load_hotpotqa, load_musique, get_all_paragraphs
from src.chunker import chunk_documents
from src.extractor import extract_triples
from src.graph_builder import build_graph, save_graph, load_graph, get_graph_stats
from src.query_engine import retrieve
from src.generator import generate_answer
from src.vanilla_rag import VanillaRAG
from src.evaluator import run_evaluation, compare_systems


def cmd_build_graph(args):
    """Build knowledge graph from dataset."""
    print(f"=== Building Knowledge Graph ({args.dataset}, {args.samples} samples) ===\n")

    # Step 1: Load data
    if args.dataset == "hotpotqa":
        records = load_hotpotqa(n_samples=args.samples)
    else:
        records = load_musique(n_samples=args.samples)
    paragraphs = get_all_paragraphs(records)
    print(f"Loaded {len(paragraphs)} unique paragraphs\n")

    # Step 2: Chunk
    chunks = chunk_documents(paragraphs)
    print(f"Created {len(chunks)} chunks\n")

    # Step 3: Extract triples
    triples = extract_triples(chunks)
    print(f"Extracted {len(triples)} triples\n")

    # Step 4: Build graph
    G = build_graph(triples)
    save_graph(G)

    # Stats
    stats = get_graph_stats(G)
    print("\n=== Graph Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def cmd_query(args):
    """Run a single query through GraphRAG."""
    G = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    result = retrieve(args.question, G)
    print(f"Query entities: {result['query_entities']}")
    print(f"Seed nodes: {result['seed_nodes']}")
    print(f"Subgraph: {result['subgraph'].number_of_nodes()} nodes, {result['subgraph'].number_of_edges()} edges")
    print(f"\nRetrieved context:\n{result['linearized']}\n")

    answer = generate_answer(args.question, result["linearized"])
    print(f"Answer: {answer}")


def cmd_evaluate(args):
    """Run evaluation comparing GraphRAG vs Vanilla RAG."""
    print(f"=== Evaluation ({args.dataset}, {args.samples} samples) ===\n")

    # Load dataset
    if args.dataset == "hotpotqa":
        records = load_hotpotqa(n_samples=args.samples)
    else:
        records = load_musique(n_samples=args.samples)
    paragraphs = get_all_paragraphs(records)
    chunks = chunk_documents(paragraphs)

    # Load or build graph
    try:
        G = load_graph()
        print(f"Loaded existing graph: {G.number_of_nodes()} nodes\n")
    except FileNotFoundError:
        print("No existing graph found. Building...")
        triples = extract_triples(chunks)
        G = build_graph(triples)
        save_graph(G)

    # Build vanilla RAG baseline
    print("Building Vanilla RAG baseline...")
    vanilla = VanillaRAG(chunks)

    # Define pipeline functions
    def graphrag_pipeline(question: str) -> dict:
        result = retrieve(question, G)
        answer = generate_answer(question, result["linearized"])
        return {"answer": answer, "retrieved_titles": result["retrieved_titles"]}

    def vanilla_pipeline(question: str) -> dict:
        return vanilla.answer(question)

    # Run evaluations
    print("\n--- Evaluating GraphRAG ---")
    graphrag_results = run_evaluation(records, graphrag_pipeline, label="GraphRAG")

    print("\n--- Evaluating Vanilla RAG ---")
    vanilla_results = run_evaluation(records, vanilla_pipeline, label="VanillaRAG")

    # Compare
    print("\n=== Results ===")
    compare_systems([graphrag_results, vanilla_results])


def cmd_stats(args):
    """Print graph statistics."""
    G = load_graph()
    stats = get_graph_stats(G)
    print("=== Knowledge Graph Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # build-graph
    p_build = subparsers.add_parser("build-graph", help="Build knowledge graph from dataset")
    p_build.add_argument("--dataset", choices=["hotpotqa", "musique"], default="hotpotqa")
    p_build.add_argument("--samples", type=int, default=config.HOTPOTQA_SAMPLES)

    # query
    p_query = subparsers.add_parser("query", help="Run a single query")
    p_query.add_argument("question", type=str)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    p_eval.add_argument("--dataset", choices=["hotpotqa", "musique"], default="hotpotqa")
    p_eval.add_argument("--samples", type=int, default=config.HOTPOTQA_SAMPLES)

    # stats
    subparsers.add_parser("stats", help="Print graph statistics")

    args = parser.parse_args()

    if args.command == "build-graph":
        cmd_build_graph(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
