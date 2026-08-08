#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SM_MODELS = {"semicrf", "treecrf"}

DATASETS = [
    {
        "name": "bn_smallmodel_mixed_163",
        "domain": "bn",
        "human_file": "annotations_wlj_smallmodel_163_final.json",
        "source_by_type": {
            "o1mini_right": "correct_data_bn_smallmodel.json",
            "o1mini_wrong": "correct_data_bn_smallmodel_deepseek.json",
        },
        "source_total_by_type": {
            "o1mini_right": "correct_data_bn_smallmodel.json",
            "o1mini_wrong": "correct_data_bn_smallmodel_deepseek.json",
        },
    },
    {
        "name": "bn_smallmodel_o1mini_wrong_deepseek_wrong_random30",
        "domain": "bn",
        "human_file": "annotations_smallmodel_botherror_random_wlj_final.json",
        "source_by_type": {
            "o1mini_wrong": "incorrect_data_bn_smallmodel_deepseek.json",
        },
        "source_total_by_type": {
            "o1mini_wrong": "incorrect_data_bn_smallmodel_deepseek.json",
        },
    },
    {
        "name": "tc_smallmodel_o1mini_right_322",
        "domain": "tc",
        "human_file": "annotation_tc_smallmodel_removerepeate_322_wlj_final.json",
        "source_by_type": {
            "o1mini_right": "correct_data_tc_smallmodel.json",
        },
        "source_total_by_type": {
            "o1mini_right": "correct_data_tc_smallmodel.json",
        },
    },
    {
        "name": "tc_smallmodel_o1mini_wrong_random30",
        "domain": "tc",
        "human_file": "annotation_tc_smallmodel_o1wrong_random30_wlj.json",
        "source_by_type": {
            "o1mini_wrong": "incorrect_data_tc_smallmodel.json",
        },
        "source_total_by_type": {
            "o1mini_wrong": "incorrect_data_tc_smallmodel.json",
        },
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


def norm_span_mean(value: Any) -> str:
    return norm_text(value).lower()


def to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def align_key(item: Dict[str, Any]) -> Tuple[str, str, int, str, str]:
    sentence = item.get("sen", item.get("sentence", ""))
    return (
        norm_text(sentence),
        str(item.get("prd_word", "")),
        to_int(item.get("prd_idx", item.get("pred.idx", -1))),
        str(item.get("label", "")),
        norm_span_mean(item.get("span_mean")),
    )


def llm_type(item: Dict[str, Any]) -> str:
    value = str(item.get("type", ""))
    if "o1mini_right" in value:
        return "o1mini_right"
    if "o1mini_wrong" in value:
        return "o1mini_wrong"
    raise ValueError(f"Cannot infer o1mini label from type={value!r}")


def llm_label(item: Dict[str, Any]) -> int:
    return 1 if llm_type(item) == "o1mini_right" else 0


def selected_spans(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    spans = item.get("selected_spans", [])
    return spans if isinstance(spans, list) else []


def human_label_smallmodel(item: Dict[str, Any]) -> int:
    for span in selected_spans(item):
        models = span.get("models", [])
        if isinstance(models, list) and any(str(model) in SM_MODELS for model in models):
            return 1
    return 0


def is_optional(item: Dict[str, Any]) -> bool:
    value = item.get("optional", False)
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1", "optional", "可标可不标"}
    return bool(value)


def cohen_kappa(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    a = b = c = d = 0.0
    for row in rows:
        llm = row["llm_label"]
        human = row["human_label"]
        weight = float(row.get("weight", 1.0))
        if llm == 1 and human == 1:
            a += weight
        elif llm == 1 and human == 0:
            b += weight
        elif llm == 0 and human == 1:
            c += weight
        elif llm == 0 and human == 0:
            d += weight
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


def source_indexes(dataset: Dict[str, Any]) -> Dict[str, Dict[Tuple[str, str, int, str, str], List[Dict[str, Any]]]]:
    indexes = {}
    for type_name, file_name in dataset["source_by_type"].items():
        idx: Dict[Tuple[str, str, int, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for item in get_items(read_json(DATA_DIR / file_name)):
            idx[align_key(item)].append(item)
        indexes[type_name] = idx
    return indexes


def source_totals(dataset: Dict[str, Any]) -> Dict[str, int]:
    totals = {}
    for type_name, file_name in dataset["source_total_by_type"].items():
        totals[type_name] = len(get_items(read_json(DATA_DIR / file_name)))
    return totals


def build_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"datasets": {}, "notes": []}

    for ds in DATASETS:
        human_items = get_items(read_json(DATA_DIR / ds["human_file"]))
        indexes = source_indexes(ds)
        totals = source_totals(ds)
        sample_counts = Counter(llm_type(item) for item in human_items)
        weights = {
            type_name: totals[type_name] / sample_counts[type_name]
            for type_name in sample_counts
            if sample_counts[type_name] > 0 and type_name in totals
        }

        unmatched = 0
        duplicate_cases = 0
        grammar_counter = Counter()
        optional_count = 0

        for item in human_items:
            typ = llm_type(item)
            label = llm_label(item)
            human = human_label_smallmodel(item)
            key = align_key(item)
            candidates = indexes.get(typ, {}).get(key, [])
            if not candidates:
                unmatched += 1
            if len(candidates) > 1:
                duplicate_cases += 1
            grammar_status = item.get("grammar_status", "")
            optional = is_optional(item)
            grammar_counter[str(grammar_status)] += 1
            optional_count += int(optional)
            rows.append(
                {
                    "dataset": ds["name"],
                    "domain": ds["domain"],
                    "human_file": ds["human_file"],
                    "source_type": typ,
                    "llm_label": label,
                    "human_label": human,
                    "llm_human_agree": int(label == human),
                    "weight": weights.get(typ, 1.0),
                    "source_total_for_type": totals.get(typ),
                    "sample_count_for_type": sample_counts.get(typ),
                    "source_matched": bool(candidates),
                    "num_source_candidates": len(candidates),
                    "human_idx": item.get("idx"),
                    "sentence": key[0],
                    "prd_word": key[1],
                    "prd_idx": key[2],
                    "label": key[3],
                    "span_mean": key[4],
                    "grammar_status": grammar_status,
                    "optional": optional,
                    "selected_spans": selected_spans(item),
                    "type": item.get("type", ""),
                }
            )

        ds_rows = [row for row in rows if row["dataset"] == ds["name"]]
        summary["datasets"][ds["name"]] = {
            "domain": ds["domain"],
            "human_file": ds["human_file"],
            "human_items": len(human_items),
            "sample_counts_by_o1mini_type": dict(sample_counts),
            "source_totals_by_o1mini_type": totals,
            "weights_by_o1mini_type": weights,
            "unmatched_human_items": unmatched,
            "duplicate_source_candidate_cases": duplicate_cases,
            "grammar_status_counts": dict(grammar_counter),
            "optional_count": optional_count,
            "kappa_unweighted": cohen_kappa([{**row, "weight": 1.0} for row in ds_rows]),
            "kappa_weighted": cohen_kappa(ds_rows),
        }

    matched_rows = [row for row in rows if row["source_matched"]]
    valid_rows = [
        row for row in matched_rows
        if row.get("grammar_status") in {"", "没有语法错误", None} and not row.get("optional")
    ]
    weighted_rows = matched_rows
    weighted_valid_rows = valid_rows

    invalid_rows = [row for row in matched_rows if row["llm_label"] == 0]
    weighted_invalid = cohen_kappa(invalid_rows)
    invalid_weight = weighted_invalid.get("llm0_human1", 0) + weighted_invalid.get("llm0_human0", 0)
    invalid_confirmed = weighted_invalid.get("llm0_human0", 0)

    summary["overall"] = {
        "matched_rows": len(matched_rows),
        "valid_rows_excluding_optional_and_grammar_errors": len(valid_rows),
        "kappa_unweighted_all_matched": cohen_kappa([{**row, "weight": 1.0} for row in matched_rows]),
        "kappa_weighted_all_matched": cohen_kappa(weighted_rows),
        "kappa_unweighted_valid_only": cohen_kappa([{**row, "weight": 1.0} for row in valid_rows]),
        "kappa_weighted_valid_only": cohen_kappa(weighted_valid_rows),
        "invalid_subset_unweighted": cohen_kappa([{**row, "weight": 1.0} for row in invalid_rows]),
        "invalid_subset_weighted": weighted_invalid,
        "weighted_human_confirmed_invalid_rate": invalid_confirmed / invalid_weight if invalid_weight else None,
    }
    summary["notes"].append("llm_label=1 means o1mini judged the small-model prediction valid; llm_label=0 means o1mini judged it invalid.")
    summary["notes"].append("human_label=1 means the final human annotation retained at least one small-model span (semicrf/treecrf); human_label=0 otherwise.")
    summary["notes"].append("Weights are source stratum size divided by the manually annotated sample size for that stratum.")
    return rows, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "dataset", "domain", "source_type", "llm_label", "human_label", "llm_human_agree", "weight",
        "source_total_for_type", "sample_count_for_type", "source_matched", "num_source_candidates",
        "human_idx", "sentence", "prd_word", "prd_idx", "label", "span_mean", "grammar_status", "optional", "type", "human_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, summary = build_rows()
    write_json(DATA_DIR / "aligned_llmj_human_smallmodel.json", rows)
    write_csv(DATA_DIR / "aligned_llmj_human_smallmodel.csv", rows)
    write_json(DATA_DIR / "smallmodel_kappa_summary.json", summary)
    print("Wrote:")
    print("  " + str(DATA_DIR / "aligned_llmj_human_smallmodel.json"))
    print("  " + str(DATA_DIR / "aligned_llmj_human_smallmodel.csv"))
    print("  " + str(DATA_DIR / "smallmodel_kappa_summary.json"))
    print("\nOverall:")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("\nPer dataset:")
    for name, info in summary["datasets"].items():
        print(name, json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
