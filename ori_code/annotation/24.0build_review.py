import argparse
import json
import random
from pathlib import Path


CORE_LABELS = [f"ARG{i}" for i in range(6)]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_key(key):
    sentence, prd_word, prd_idx, label = key.split("\t")
    return sentence, prd_word, int(prd_idx), label


def is_grammar_error(item):
    return (
        item.get("grammar_status") == "有语法错误"
        or item.get("grammar_error_desc") == "有语法错误"
    )


def normalize_span(span, source="candidate"):
    return {
        "start": span.get("start"),
        "end": span.get("end"),
        "text": span.get("text", ""),
        "models": span.get("models", []),
        "source": span.get("source", source),
    }


def group_final_data(data, keep_grammar_errors=False):
    grouped = {}
    for key, item in data.items():
        sentence, prd_word, prd_idx, label = parse_key(key)
        if label not in CORE_LABELS:
            continue
        if is_grammar_error(item) and not keep_grammar_errors:
            continue

        pred_key = (sentence, prd_word, prd_idx)
        if pred_key not in grouped:
            grouped[pred_key] = {
                "sentence": sentence,
                "prd_word": prd_word,
                "prd_idx": prd_idx,
                "labels": {},
            }

        candidate_spans = [
            normalize_span(span, source="candidate")
            for span in item.get("selected_spans", [])
        ]
        grouped[pred_key]["labels"][label] = {
            "label": label,
            "role_description": item.get("span_mean"),
            "optional": bool(item.get("optional", False)),
            "grammar_status": item.get("grammar_status", ""),
            "grammar_error_desc": item.get("grammar_error_desc", ""),
            "candidate_spans": candidate_spans,
            # Review starts from current corrected result. Reviewers can remove,
            # keep, or add spans. Added spans are marked source=human_added.
            "reviewed_spans": candidate_spans.copy(),
            "no_valid_span": len(candidate_spans) == 0,
            "review_note": "",
        }

    # for pred in grouped.values():
    #     for label in CORE_LABELS:
    #         pred["labels"].setdefault(
    #             label,
    #             {
    #                 "label": label,
    #                 "role_description": None,
    #                 "optional": False,
    #                 "grammar_status": "",
    #                 "grammar_error_desc": "",
    #                 "candidate_spans": [],
    #                 "reviewed_spans": [],
    #                 "no_valid_span": True,
    #                 "review_note": "",
    #             },
    #         )
    return grouped


def sample_predicates(grouped, sample_size, seed):
    keys = sorted(grouped.keys(), key=str)
    if sample_size is None:
        selected = keys
    else:
        rng = random.Random(seed)
        selected = rng.sample(keys, min(sample_size, len(keys)))
    return [grouped[key] for key in selected]


def build_review_file(domain_to_path, sample_size_per_domain, seed):
    records = []
    settings = {
        "core_labels": CORE_LABELS,
        "sample_size_per_domain": sample_size_per_domain,
        "seed": seed,
        "recall_definition": (
            "span_recall = human-reviewed valid spans already present in "
            "candidate_spans / all human-reviewed valid spans"
        ),
    }

    for offset, (domain, path) in enumerate(domain_to_path.items()):
        grouped = group_final_data(read_json(path))
        sampled = sample_predicates(grouped, sample_size_per_domain, seed + offset)
        for i, pred in enumerate(sampled, start=1):
            records.append(
                {
                    "review_id": f"{domain}_{i:03d}",
                    "domain": domain,
                    "sentence": pred["sentence"],
                    "prd_word": pred["prd_word"],
                    "prd_idx": pred["prd_idx"],
                    # "labels": [pred["labels"][label] for label in CORE_LABELS],
                    # Only keep labels that already exist in the corrected data.
                    # If an existing label has no selected spans, it is still
                    # retained with no_valid_span=True.
                    "labels": [
                        pred["labels"][label]
                        for label in sorted(
                            pred["labels"],
                            key=lambda x: CORE_LABELS.index(x) if x in CORE_LABELS else 999,
                        )
                    ],
                    "review_status": "pending",
                }
            )

    return {"settings": settings, "records": records}


def main():
    parser = argparse.ArgumentParser(
        description="Build a predicate-level review file for span-recall estimation."
    )
    parser.add_argument("--bn-file", default="analysis/test_bn_500_core_final_v4.json")
    parser.add_argument("--tc-file", default="analysis/test_tc_core_final_v2.json")
    parser.add_argument(
        "--sample-size-per-domain",
        type=int,
        default=25,
        help="Use -1 to include all predicates.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="analysis/span_recall_review_50.json")
    args = parser.parse_args()

    sample_size = None if args.sample_size_per_domain < 0 else args.sample_size_per_domain
    review = build_review_file(
        {"bn": args.bn_file, "tc": args.tc_file},
        sample_size_per_domain=sample_size,
        seed=args.seed,
    )
    write_json(Path(args.out), review)
    print(f"Review records: {len(review['records'])}")
    print(f"Written to: {args.out}")


if __name__ == "__main__":
    main()
