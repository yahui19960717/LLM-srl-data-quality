#!/usr/bin/env python3
"""Align o1-mini LLM-J judgements on gold data with human annotations.

The script builds binary labels for later Cohen kappa computation:
  llm_label   = 1 if o1-mini judged the original gold span-label as correct,
                0 if o1-mini judged it as incorrect.
  human_label = 1 if the human final annotation keeps the exact same original
                gold span, 0 otherwise.

Human annotation files often reset idx inside each annotation subset, so this
script aligns by (sentence, prd_word, prd_idx, label), while keeping both idx
fields in the output for inspection.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASETS = [
    {
        "name": "bn_gold_o1mini_correct",
        "domain": "bn",
        "llm_file": "correct_data_bn_gold.json",
        "human_file": "annotations_gold_o1right_random_wlj_final.json",
        "llm_label": 1,
        "llm_judgement": "correct",
    },
    {
        "name": "tc_gold_o1mini_correct",
        "domain": "tc",
        "llm_file": "correct_data_tc_gold.json",
        "human_file": "annotation_tc_gold_o1right_random30_wlj.json",
        "llm_label": 1,
        "llm_judgement": "correct",
    },
    {
        "name": "bn_gold_o1mini_incorrect",
        "domain": "bn",
        "llm_file": "incorrect_data_bn_gold.json",
        "human_file": "annotations_single_gold_o1wrongdsright_93_wlj_final.json",
        "llm_label": 0,
        "llm_judgement": "incorrect",
    },
    {
        "name": "tc_gold_o1mini_incorrect",
        "domain": "tc",
        "llm_file": "incorrect_data_tc_gold.json",
        "human_file": "annotation_tc_single_gold_201_wlj_final.json",
        "llm_label": 0,
        "llm_judgement": "incorrect",
    },
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_items(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    if isinstance(data, list):
        return data
    raise ValueError("JSON must be a list or contain an 'annotations' field")


def norm_sentence(text: Any) -> str:
    return " ".join(str(text or "").split())


def to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def align_key(item: Dict[str, Any]) -> Tuple[str, str, int, str]:
    sentence = item.get("sen", item.get("sentence", ""))
    return (
        norm_sentence(sentence),
        str(item.get("prd_word", "")),
        to_int(item.get("prd_idx", item.get("pred.idx", -1))),
        str(item.get("label", "")),
    )


def llm_span_inclusive(item: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    span_idx = item.get("span_idx")
    if not isinstance(span_idx, list) or len(span_idx) != 2:
        return None
    return (to_int(span_idx[0]), to_int(span_idx[1]) - 1)


def selected_spans(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    spans = item.get("selected_spans", [])
    return spans if isinstance(spans, list) else []


def selected_span_set(item: Dict[str, Any]) -> set[Tuple[int, int]]:
    spans = set()
    for span in selected_spans(item):
        if "start" in span and "end" in span:
            spans.add((to_int(span["start"]), to_int(span["end"])))
    return spans


def has_gold_model(item: Dict[str, Any]) -> bool:
    for span in selected_spans(item):
        models = span.get("models", [])
        if isinstance(models, list) and "gold" in models:
            return True
    return False


def choose_llm_candidate(
    human_item: Dict[str, Any], candidates: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    human_spans = selected_span_set(human_item)
    human_idx = human_item.get("idx")

    def score(candidate: Dict[str, Any]) -> Tuple[int, int]:
        span = llm_span_inclusive(candidate)
        span_match = int(span in human_spans) if span is not None else 0
        idx_match = int(candidate.get("idx") == human_idx)
        return (span_match, idx_match)

    return sorted(candidates, key=score, reverse=True)[0]


def is_optional(item: Dict[str, Any]) -> bool:
    value = item.get("optional", False)
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1", "optional", "可标可不标"}
    return bool(value)


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
        "llm1_human1": a,
        "llm1_human0": b,
        "llm0_human1": c,
        "llm0_human0": d,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def build_alignment() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"datasets": {}, "notes": []}

    for ds in DATASETS:
        llm_path = DATA_DIR / ds["llm_file"]
        human_path = DATA_DIR / ds["human_file"]
        llm_items = get_items(read_json(llm_path))
        human_items = get_items(read_json(human_path))

        llm_index: Dict[Tuple[str, str, int, str], List[Dict[str, Any]]] = defaultdict(list)
        for item in llm_items:
            llm_index[align_key(item)].append(item)

        matched = 0
        unmatched = 0
        duplicate_candidate_cases = 0
        span_match_count = 0
        gold_model_count = 0
        grammar_counter = Counter()
        optional_count = 0

        for human_item in human_items:
            key = align_key(human_item)
            candidates = llm_index.get(key, [])
            if not candidates:
                unmatched += 1
                rows.append(
                    {
                        "dataset": ds["name"],
                        "domain": ds["domain"],
                        "status": "unmatched_human_item",
                        "human_idx": human_item.get("idx"),
                        "llm_idx": None,
                        "sentence": key[0],
                        "prd_word": key[1],
                        "prd_idx": key[2],
                        "label": key[3],
                        "llm_label": ds["llm_label"],
                        "human_label": None,
                    }
                )
                continue

            matched += 1
            if len(candidates) > 1:
                duplicate_candidate_cases += 1
            llm_item = choose_llm_candidate(human_item, candidates)
            assert llm_item is not None

            llm_span = llm_span_inclusive(llm_item)
            human_spans = selected_span_set(human_item)
            span_aligned = llm_span in human_spans if llm_span is not None else False
            gold_model = has_gold_model(human_item)
            human_label = 1 if span_aligned else 0
            agree = int(ds["llm_label"] == human_label)
            grammar_status = human_item.get("grammar_status", "")
            optional = is_optional(human_item)

            span_match_count += int(span_aligned)
            gold_model_count += int(gold_model)
            grammar_counter[str(grammar_status)] += 1
            optional_count += int(optional)

            rows.append(
                {
                    "dataset": ds["name"],
                    "domain": ds["domain"],
                    "status": "matched",
                    "llm_judgement": ds["llm_judgement"],
                    "llm_label": ds["llm_label"],
                    "human_label": human_label,
                    "llm_human_agree": agree,
                    "span_aligned": span_aligned,
                    "human_has_gold_model": gold_model,
                    "num_llm_candidates_for_key": len(candidates),
                    "llm_idx": llm_item.get("idx"),
                    "human_idx": human_item.get("idx"),
                    "idx_equal": llm_item.get("idx") == human_item.get("idx"),
                    "sentence": key[0],
                    "prd_word": key[1],
                    "prd_idx": key[2],
                    "label": key[3],
                    "llm_span_start": llm_span[0] if llm_span else None,
                    "llm_span_end": llm_span[1] if llm_span else None,
                    "llm_span_text": llm_item.get("span"),
                    "human_selected_spans": selected_spans(human_item),
                    "grammar_status": grammar_status,
                    "optional": optional,
                    "llm_file": ds["llm_file"],
                    "human_file": ds["human_file"],
                }
            )

        ds_rows = [
            r for r in rows if r.get("dataset") == ds["name"] and r.get("status") == "matched"
        ]
        summary["datasets"][ds["name"]] = {
            "domain": ds["domain"],
            "llm_file": ds["llm_file"],
            "human_file": ds["human_file"],
            "llm_items": len(llm_items),
            "human_items": len(human_items),
            "matched_human_items": matched,
            "unmatched_human_items": unmatched,
            "duplicate_candidate_cases": duplicate_candidate_cases,
            "span_aligned_count": span_match_count,
            "human_has_gold_model_count": gold_model_count,
            "grammar_status_counts": dict(grammar_counter),
            "optional_count": optional_count,
            "kappa_on_matched_rows": cohen_kappa(ds_rows),
        }

    matched_rows = [r for r in rows if r.get("status") == "matched"]
    valid_rows = [
        r
        for r in matched_rows
        if r.get("grammar_status") in {"", "没有语法错误", None} and not r.get("optional")
    ]
    summary["overall"] = {
        "matched_rows": len(matched_rows),
        "valid_rows_excluding_optional_and_grammar_errors": len(valid_rows),
        "kappa_all_matched": cohen_kappa(matched_rows),
        "kappa_valid_only": cohen_kappa(valid_rows),
    }
    summary["notes"].append(
        "human_label=1 means the human final annotation keeps the exact LLM/original gold span."
    )
    summary["notes"].append(
        "Alignment uses (sentence, prd_word, prd_idx, label), because human annotation idx can be reset inside each subset."
    )
    return rows, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "dataset",
        "domain",
        "status",
        "llm_judgement",
        "llm_label",
        "human_label",
        "llm_human_agree",
        "span_aligned",
        "human_has_gold_model",
        "num_llm_candidates_for_key",
        "llm_idx",
        "human_idx",
        "idx_equal",
        "sentence",
        "prd_word",
        "prd_idx",
        "label",
        "llm_span_start",
        "llm_span_end",
        "llm_span_text",
        "grammar_status",
        "optional",
        "llm_file",
        "human_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, summary = build_alignment()
    write_json(DATA_DIR / "aligned_llmj_human_gold.json", rows)
    write_csv(DATA_DIR / "aligned_llmj_human_gold.csv", rows)
    write_json(DATA_DIR / "alignment_summary.json", summary)

    print("Wrote:")
    print(f"  {DATA_DIR / 'aligned_llmj_human_gold.json'}")
    print(f"  {DATA_DIR / 'aligned_llmj_human_gold.csv'}")
    print(f"  {DATA_DIR / 'alignment_summary.json'}")
    print("\nOverall:")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("\nPer dataset:")
    for name, info in summary["datasets"].items():
        print(name, json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
