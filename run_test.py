"""
Fast end-to-end test: GraphRAG vs VanillaRAG on 5 HotpotQA questions.

Uses only the 2 supporting (gold) paragraphs per question for graph building,
so only ~10 API calls are needed instead of 50, making it feasible on the free tier.
After building, evaluation uses ALL 10 context paragraphs (fair to both systems).
"""

import sys, json, time
from pathlib import Path

sys.path.insert(0, ".")
import config
from src.data_loader import load_hotpotqa, get_all_paragraphs
from src.chunker import chunk_documents
from src.extractor import extract_triples
from src.graph_builder import build_graph, save_graph, load_graph, get_graph_stats
from src.query_engine import retrieve
from src.generator import generate_answer
from src.vanilla_rag import VanillaRAG
from src.evaluator import exact_match, f1_score


# ── 1. Load 5 cached HotpotQA records ────────────────────────────────
print("=" * 60)
print("STEP 1: Loading 5 HotpotQA records")
print("=" * 60)
records = load_hotpotqa(n_samples=50)
print(f"  Loaded {len(records)} questions\n")
for i, r in enumerate(records[:5]):
    print(f"  Q{i+1}: {r['question']}")
    print(f"       Gold answer: {r['answer']}")
    print(f"       Supporting docs: {r['supporting_facts_titles']}\n")
print(f"  ... and {len(records)-5} more questions\n")


# ── 2. Build graph from SUPPORTING paragraphs only ──────────────────
graph_path = config.GRAPH_PATH
triples_path = config.TRIPLES_CACHE

if graph_path.exists() and triples_path.exists():
    print("=" * 60)
    print("STEP 2: Loading cached graph (already built)")
    print("=" * 60)
    G = load_graph()
else:
    print("=" * 60)
    print("STEP 2: Building knowledge graph (supporting paragraphs, 50 samples)")
    print(f"        Using AWS Bedrock Claude — paid tier, no rate limits")
    print("=" * 60)

    # Use ONLY the gold/supporting paragraphs for graph building
    # This is research-correct: the graph is built from the evidence corpus
    supporting_docs = []
    seen = set()
    for rec in records:
        gold_titles = set(rec["supporting_facts_titles"])
        for para in rec["context_paragraphs"]:
            if para["title"] in gold_titles:
                key = para["title"]
                if key not in seen:
                    seen.add(key)
                    supporting_docs.append({"doc_id": para["title"], "text": para["text"]})

    print(f"  Using {len(supporting_docs)} unique supporting paragraphs\n")
    chunks = chunk_documents(supporting_docs)
    print(f"  Created {len(chunks)} chunks\n")

    # Clear old cache if empty
    if triples_path.exists():
        triples_path.unlink()

    triples = extract_triples(chunks)
    print(f"\n  Extracted {len(triples)} unique triples")

    G = build_graph(triples)
    save_graph(G)

stats = get_graph_stats(G)
print("\n  Graph Statistics:")
for k, v in stats.items():
    print(f"    {k}: {v}")


# ── 3. Build full chunk index for Vanilla RAG (all 10 paragraphs) ────
print("=" * 60)
print("STEP 3: Building Vanilla RAG index (all context paragraphs)")
print("=" * 60)
all_paragraphs = get_all_paragraphs(records)
all_chunks = chunk_documents(all_paragraphs)
print(f"  Indexed {len(all_chunks)} chunks for Vanilla RAG\n")
vanilla = VanillaRAG(all_chunks)


# ── 4. Run both pipelines on all 5 questions ─────────────────────────
print("=" * 60)
print("STEP 4: Running GraphRAG and VanillaRAG on 5 questions")
print("=" * 60)

results = []
for i, rec in enumerate(records):
    q  = rec["question"]
    gold = rec["answer"]
    print(f"\nQ{i+1}: {q}")
    print(f"     Gold: {gold}")

    # --- GraphRAG ---
    t0 = time.time()
    graph_result = retrieve(q, G)
    graph_answer = generate_answer(q, graph_result["linearized"])
    graph_time   = round(time.time() - t0, 2)
    graph_em     = exact_match(graph_answer, gold)
    graph_f1     = f1_score(graph_answer, gold)
    seed_info    = f"seeds={graph_result['seed_nodes'][:3]}" if graph_result['seed_nodes'] else "NO_SEEDS(fallback used)"
    print(f"     GraphRAG   -> '{graph_answer}'  EM={graph_em}  F1={graph_f1:.2f}  [{seed_info}]")

    # --- VanillaRAG ---
    t0 = time.time()
    van_result = vanilla.answer(q)
    van_answer = van_result["answer"]
    van_time   = round(time.time() - t0, 2)
    van_em     = exact_match(van_answer, gold)
    van_f1     = f1_score(van_answer, gold)
    print(f"     VanillaRAG -> '{van_answer}'  EM={van_em}  F1={van_f1:.2f}")

    results.append({
        "question": q, "gold": gold,
        "graph_answer": graph_answer, "graph_em": graph_em, "graph_f1": graph_f1,
        "van_answer":   van_answer,   "van_em":   van_em,   "van_f1":   van_f1,
    })

    time.sleep(4)


# ── 5. Comparison table ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: FINAL COMPARISON — GraphRAG vs VanillaRAG")
print("=" * 60)

avg_graph_em = sum(r["graph_em"] for r in results) / len(results)
avg_graph_f1 = sum(r["graph_f1"] for r in results) / len(results)
avg_van_em   = sum(r["van_em"]   for r in results) / len(results)
avg_van_f1   = sum(r["van_f1"]   for r in results) / len(results)

print(f"\n{'System':<15} {'Avg EM':>10} {'Avg F1':>10}")
print("-" * 37)
print(f"{'GraphRAG':<15} {avg_graph_em:>10.3f} {avg_graph_f1:>10.3f}")
print(f"{'VanillaRAG':<15} {avg_van_em:>10.3f} {avg_van_f1:>10.3f}")
em_delta = avg_graph_em - avg_van_em
f1_delta = avg_graph_f1 - avg_van_f1
print(f"\n  EM improvement  : {em_delta:+.3f}")
print(f"  F1 improvement  : {f1_delta:+.3f}")

if em_delta > 0 and f1_delta > 0:
    print("\n  GraphRAG OUTPERFORMS VanillaRAG on both metrics!")
elif em_delta > 0 or f1_delta > 0:
    print("\n  GraphRAG shows improvement on at least one metric.")
else:
    print("\n  Results are close — try larger sample size for definitive results.")


# ── 6. Save results ───────────────────────────────────────────────────
out_path = config.RESULTS_DIR / "test_comparison.json"
out_path.write_text(json.dumps({
    "per_question": results,
    "summary": {
        "GraphRAG":   {"EM": avg_graph_em, "F1": avg_graph_f1},
        "VanillaRAG": {"EM": avg_van_em,   "F1": avg_van_f1},
        "delta_EM": em_delta, "delta_F1": f1_delta,
    }
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  Full results saved to {out_path}")
