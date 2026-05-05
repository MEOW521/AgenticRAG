"""将不同格式的评测样本统一成内部结构。"""
import json
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_eval_item(item: dict) -> dict[str, Any]:
    """兼容两种格式：

    1) multihop 合成导出（domain_multihup_synthesis 等）::
       final_question, final_answer, answer_aliases?, hops[].doc_chunk_id

    2) 手写 flat 评测集::
       question, answer, supporting_chunk_ids?
    """
    if "final_question" in item:
        golds: list[str] = []
        fa = item.get("final_answer")
        if fa:
            golds.append(str(fa).strip())
        for a in item.get("answer_aliases") or []:
            if a is None:
                continue
            s = str(a).strip()
            if s and s not in golds:
                golds.append(s)

        chunk_ids: list[str] = []
        for h in item.get("hops") or []:
            cid = h.get("doc_chunk_id") or h.get("chunk_id")
            if cid:
                chunk_ids.append(cid)

        return {
            "id": item.get("id"),
            "question": item["final_question"],
            "gold_answers": golds,
            "supporting_chunk_ids": chunk_ids,
            "subset": item.get("subset"),
            "hop_count": item.get("hop_count"),
            "qa_type": item.get("qa_type"),
            "source": "multihop_export",
        }

    q = item.get("question", "")
    ans = item.get("answer")
    golds = [str(ans).strip()] if ans else []
    chunk_ids = list(item.get("supporting_chunk_ids") or [])
    return {
        "id": item.get("id"),
        "question": q,
        "gold_answers": golds,
        "supporting_chunk_ids": chunk_ids,
        "subset": item.get("question_type") or item.get("subset"),
        "hop_count": item.get("hop_count"),
        "qa_type": item.get("qa_type"),
        "source": "flat",
    }
