"""单次 RAG（simple_rag）评测。支持 flat jsonl 与 multihop_results.jsonl 导出格式。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
for _p in (SRC_ROOT, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.data import normalize_eval_item, read_jsonl
from eval.metrics import gold_exact_match, gold_token_f1, hit_at_k, safe_avg, summarize_by_subset


def _collect_pred_chunk_ids(evidence: list[dict]) -> list[str]:
    chunk_ids: list[str] = []
    for step in evidence:
        for r in step.get("results", []):
            cid = r.get("chunk_id")
            if cid:
                chunk_ids.append(cid)
    return chunk_ids


def run_eval(input_path: str, output_path: str, model: str | None = None) -> dict:
    items = read_jsonl(input_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        from agents.synthesizer import simple_rag
    except ImportError as e:
        raise ImportError(
            "无法导入 agents.synthesizer.simple_rag，请确保已恢复完整 agent 代码。"
        ) from e

    em_scores: list[float] = []
    f1_scores: list[float] = []
    hit_scores: list[float] = []
    tool_calls: list[float] = []
    latencies: list[float] = []
    preds: list[dict] = []

    for idx, raw in enumerate(items, start=1):
        ev = normalize_eval_item(raw)
        question = ev["question"]
        gold_answers = ev["gold_answers"]
        gold_chunks = ev["supporting_chunk_ids"]

        t0 = time.time()
        state = simple_rag({"query": question})
        latency = time.time() - t0

        pred_answer = (state.get("final_answer") or "").strip()
        evidence = state.get("evidence", [])
        pred_chunks = _collect_pred_chunk_ids(evidence)

        em = gold_exact_match(pred_answer, gold_answers)
        f1 = gold_token_f1(pred_answer, gold_answers)
        h = hit_at_k(pred_chunks, gold_chunks) if gold_chunks else 0.0

        em_scores.append(em)
        f1_scores.append(f1)
        hit_scores.append(h)
        tool_calls.append(float(state.get("total_tool_calls", 0)))
        latencies.append(latency)

        preds.append(
            {
                "id": ev.get("id") or raw.get("id") or idx,
                "subset": ev.get("subset"),
                "hop_count": ev.get("hop_count"),
                "qa_type": ev.get("qa_type"),
                "source": ev.get("source"),
                "question": question,
                "gold_answers_head": gold_answers[:3],
                "supporting_chunk_ids": gold_chunks,
                "pred_answer": pred_answer,
                "pred_chunk_ids": pred_chunks[:15],
                "em": em,
                "f1": f1,
                "hit_at_k": h,
                "latency_sec": latency,
                "tool_calls": state.get("total_tool_calls", 0),
            }
        )

    summary = {
        "num_samples": len(items),
        "input_format": "multihop_export_or_flat",
        "metrics": {
            "em": safe_avg(em_scores),
            "f1": safe_avg(f1_scores),
            "hit_at_k": safe_avg(hit_scores),
            "avg_tool_calls": safe_avg(tool_calls),
            "avg_latency_sec": safe_avg(latencies),
        },
        "by_subset": summarize_by_subset(
            [
                {"subset": p.get("subset"), "em": p["em"], "f1": p["f1"], "hit_at_k": p["hit_at_k"]}
                for p in preds
            ]
        ),
    }
    if model:
        summary["model"] = model

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": preds}, f, ensure_ascii=False, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Baseline (single-pass RAG) evaluation.")
    parser.add_argument(
        "--input",
        required=True,
        help="jsonl：flat 或 multihop_results（final_question / final_answer / answer_aliases / hops）",
    )
    parser.add_argument("--output", default="data/results/baseline_results.json")
    parser.add_argument("--model", default=None, help="仅写入 summary 元信息（若 simple_rag 内部用 env 配置模型）")
    args = parser.parse_args()

    summary = run_eval(args.input, args.output, model=args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
