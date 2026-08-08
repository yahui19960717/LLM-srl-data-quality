#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute LLM-J vs. human Cohen kappa on the review audit set.

review 目录表示“需要人工审核”的样本，包含两类 LLM-J 判断：
1. LLM-J 判断 gold span 为 invalid，记为 LLM-J label = 0。
2. LLM-J 判断 small model prediction span 为 valid，记为 LLM-J label = 1。

human 标签转换方式与 non_review 一致：
先在对应的 correct_data / incorrect_data 源文件中找到人工 final 样本对应的源样本，
然后检查源样本的 span 是否出现在 final 的 selected_spans 中。

- human_label = 1：源文件中的目标 span 出现在 final selected_spans 中，说明人保留该 span。
- human_label = 0：源文件中的目标 span 没有出现在 final selected_spans 中，说明人没有保留该 span。

注意：
源文件 span_idx 使用 [start, end) 格式，右边界不包含；
人工 final selected_spans 使用 start/end 格式，右边界包含。
因此比较前需要把源 span_idx 转为 inclusive span: (start, end - 1)。

四格矩阵定义：
a = LLM-J=1, human=1
b = LLM-J=1, human=0
c = LLM-J=0, human=1
d = LLM-J=0, human=0
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "review"

DATASETS = [
    {
        "name": "bn_gold_llmj_invalid",
        "domain": "bn",
        "llm_label": 0,
        "human_file": "annotations_single_gold_wlj_final.json",
        "source_file": "incorrect_data_bn_gold.json",
        "type_filter_contains": None,
    },
    {
        "name": "tc_gold_llmj_invalid",
        "domain": "tc",
        "llm_label": 0,
        "human_file": "annotation_tc_single_gold_201_wlj_final.json",
        "source_file": "incorrect_data_tc_gold.json",
        "type_filter_contains": None,
    },
    {
        "name": "bn_smallmodel_llmj_valid",
        "domain": "bn",
        "llm_label": 1,
        "human_file": "annotations_wlj_smallmodel_163_final.json",
        "source_file": "correct_data_bn_smallmodel.json",
        # This BN final file also contains 36 o1mini_wrong/deepseek_right items.
        # They are not part of the “LLM-J judged small model valid” stratum, so filter them out.
        "type_filter_contains": "o1mini_right",
    },
    {
        "name": "tc_smallmodel_llmj_valid",
        "domain": "tc",
        "llm_label": 1,
        "human_file": "annotation_tc_smallmodel_removerepeate_322_wlj_final.json",
        "source_file": "correct_data_tc_smallmodel.json",
        "type_filter_contains": None,
    },
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    if isinstance(data, list):
        return data
    raise ValueError("JSON must be a list or contain an annotations field")


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def align_key(item: Dict[str, Any]) -> Tuple[str, str, int, str, str]:
    """Use content fields rather than idx because annotation idx may be reset."""
    sentence = item.get("sen", item.get("sentence", ""))
    return (
        norm_text(sentence),
        str(item.get("prd_word", "")),
        to_int(item.get("prd_idx", item.get("pred.idx", -1))),
        str(item.get("label", "")),
        norm_text(item.get("span_mean", "")),
    )


def source_span_inclusive(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Convert source span_idx [start, end) to inclusive (start, end - 1)."""
    span_idx = item.get("span_idx")
    if not isinstance(span_idx, list) or len(span_idx) != 2:
        return None
    return (to_int(span_idx[0]), to_int(span_idx[1]) - 1)


def selected_span_set(item: Dict[str, Any]) -> set[Tuple[int, int]]:
    """Use all final selected spans. If any one matches the source span, human_label=1."""
    spans = set()
    selected = item.get("selected_spans", [])
    if not isinstance(selected, list):
        return spans
    for span in selected:
        if isinstance(span, dict) and "start" in span and "end" in span:
            spans.add((to_int(span["start"]), to_int(span["end"])))
    return spans


def is_optional(item: Dict[str, Any]) -> bool:
    value = item.get("optional", False)
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1", "optional", "可标可不标"}
    return bool(value)


def span_texts(item: Dict[str, Any]) -> List[str]:
    selected = item.get("selected_spans", [])
    if not isinstance(selected, list):
        return []
    return [str(span.get("text", "")) for span in selected if isinstance(span, dict)]


def cohen_kappa(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    a = b = c = d = 0
    for row in rows:
        llm = row["llm_label"]
        human = row["human_label"]
        if llm == 1 and human == 1:
            a += 1
        elif llm == 1 and human == 0:
            b += 1
        elif llm == 0 and human == 1:
            c += 1
        elif llm == 0 and human == 0:
            d += 1

    n = a + b + c + d
    if n == 0:
        return {"n": 0, "kappa": None}

    observed = (a + d) / n
    expected = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    kappa = (observed - expected) / (1 - expected) if expected != 1 else None
    return {
        "n": n,
        "matrix": {
            "llm1_human1_a": a,
            "llm1_human0_b": b,
            "llm0_human1_c": c,
            "llm0_human0_d": d,
        },
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def filter_human_items(items: List[Dict[str, Any]], contains: Optional[str]) -> List[Dict[str, Any]]:
    if not contains:
        return items
    return [item for item in items if contains in str(item.get("type", ""))]


def build_alignment() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"datasets": {}, "notes": []}

    for ds in DATASETS:
        raw_human_items = get_items(read_json(DATA_DIR / ds["human_file"]))
        human_items = filter_human_items(raw_human_items, ds.get("type_filter_contains"))
        source_items = get_items(read_json(DATA_DIR / ds["source_file"]))

        source_index: Dict[Tuple[str, str, int, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for source_item in source_items:
            source_index[align_key(source_item)].append(source_item)

        unmatched = 0
        duplicate_source_cases = 0
        grammar_counter = Counter()
        optional_count = 0
        dataset_rows: List[Dict[str, Any]] = []

        for human_item in human_items:
            key = align_key(human_item)
            candidates = source_index.get(key, [])
            if not candidates:
                unmatched += 1
            if len(candidates) > 1:
                duplicate_source_cases += 1

            candidate_spans = [source_span_inclusive(item) for item in candidates]
            candidate_spans = [span for span in candidate_spans if span is not None]
            human_spans = selected_span_set(human_item)

            # If there are multiple source candidates with the same key, any exact span match counts.
            span_matched = any(span in human_spans for span in candidate_spans)
            human_label = 1 if span_matched else 0

            grammar_status = human_item.get("grammar_status", "")
            optional = is_optional(human_item)
            grammar_counter[str(grammar_status)] += 1
            optional_count += int(optional)

            row = {
                "dataset": ds["name"],
                "domain": ds["domain"],
                "llm_label": ds["llm_label"],
                "human_label": human_label,
                "llm_human_agree": int(ds["llm_label"] == human_label),
                "status": "matched" if candidates else "unmatched_human_item",
                "num_source_candidates": len(candidates),
                "human_idx": human_item.get("idx"),
                "source_idxs": [item.get("idx") for item in candidates],
                "sentence": key[0],
                "prd_word": key[1],
                "prd_idx": key[2],
                "label": key[3],
                "span_mean": key[4],
                "source_spans_inclusive": candidate_spans,
                "human_selected_spans": sorted(human_spans),
                "human_selected_texts": span_texts(human_item),
                "span_matched": span_matched,
                "grammar_status": grammar_status,
                "optional": optional,
                "type": human_item.get("type", ""),
                "human_file": ds["human_file"],
                "source_file": ds["source_file"],
            }
            rows.append(row)
            dataset_rows.append(row)

        matched_dataset_rows = [row for row in dataset_rows if row["status"] == "matched"]
        summary["datasets"][ds["name"]] = {
            "domain": ds["domain"],
            "llm_label": ds["llm_label"],
            "human_file": ds["human_file"],
            "source_file": ds["source_file"],
            "raw_human_items": len(raw_human_items),
            "used_human_items_after_type_filter": len(human_items),
            "type_filter_contains": ds.get("type_filter_contains"),
            "source_items": len(source_items),
            "matched_human_items": len(matched_dataset_rows),
            "unmatched_human_items": unmatched,
            "duplicate_source_candidate_cases": duplicate_source_cases,
            "grammar_status_counts": dict(grammar_counter),
            "optional_count": optional_count,
            "human_label_counts": dict(Counter(row["human_label"] for row in matched_dataset_rows)),
            "matrix_and_kappa": cohen_kappa(matched_dataset_rows),
        }

    matched_rows = [row for row in rows if row["status"] == "matched"]
    valid_rows = [
        row for row in matched_rows
        if row.get("grammar_status") in {"", "没有语法错误", None} and not row.get("optional")
    ]

    summary["overall"] = {
        "matched_rows": len(matched_rows),
        "valid_rows_excluding_optional_and_grammar_errors": len(valid_rows),
        "matrix_and_kappa_all_matched": cohen_kappa(matched_rows),
        "matrix_and_kappa_valid_only": cohen_kappa(valid_rows),
    }
    summary["notes"].append("LLM-J label 0 means LLM-J judged the gold span invalid.")
    summary["notes"].append("LLM-J label 1 means LLM-J judged the small model prediction span valid.")
    summary["notes"].append("Human label 1 means the source span appears in final selected_spans; human label 0 means it does not.")
    summary["notes"].append("Source span_idx is converted from [start, end) to inclusive (start, end - 1) before comparison.")
    summary["notes"].append("BN smallmodel final contains mixed types; only o1mini_right items are used for the review smallmodel-valid stratum.")
    return rows, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "domain",
        "llm_label",
        "human_label",
        "llm_human_agree",
        "status",
        "num_source_candidates",
        "human_idx",
        "source_idxs",
        "sentence",
        "prd_word",
        "prd_idx",
        "label",
        "span_mean",
        "source_spans_inclusive",
        "human_selected_spans",
        "span_matched",
        "grammar_status",
        "optional",
        "type",
        "human_file",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, summary = build_alignment()
    write_json(DATA_DIR / "aligned_review_kappa.json", rows)
    write_csv(DATA_DIR / "aligned_review_kappa.csv", rows)
    write_json(DATA_DIR / "review_kappa_summary.json", summary)

    print("Wrote:")
    print("  " + str(DATA_DIR / "aligned_review_kappa.json"))
    print("  " + str(DATA_DIR / "aligned_review_kappa.csv"))
    print("  " + str(DATA_DIR / "review_kappa_summary.json"))
    print("\nOverall:")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("\nPer dataset:")
    for name, info in summary["datasets"].items():
        print(name, json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
