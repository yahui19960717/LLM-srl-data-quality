import argparse
import json
from collections import defaultdict
from pathlib import Path


# 文件名修改这里即可；也可以在命令行用参数覆盖。
DEFAULT_REVIEWER_A = "analysis/span_recall_review_50_reviewed_lyh.json"
DEFAULT_REVIEWER_B = "analysis/span_recall_review_50_reviewed_wlj.json"
DEFAULT_DISAGREEMENTS = "analysis/span_recall_review_disagreements.json"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_key(record):
    """对齐两个 reviewer 的同一个谓词。"""
    return (
        record.get("domain", ""),
        record.get("sentence", ""),
        record.get("prd_word", ""),
        str(record.get("prd_idx", "")),
    )


def label_key(record, label):
    """span-level 统计时的 predicate-role key。"""
    return record_key(record) + (label.get("label", ""),)


def span_key(record, label, span):
    """
    span-level key。
    multiple valid spans 要逐个计数，所以 start/end/text 都作为 span 身份。
    如果你只想按边界比较，可以把 text 从这里删掉。
    """
    return label_key(record, label) + (
        str(span.get("start", "")),
        str(span.get("end", "")),
        span.get("text", ""),
    )


def simple_span_key(span):
    """只用于判断一个最终 span 是否在同 label 的 candidate_spans 中。"""
    return (str(span.get("start", "")), str(span.get("end", "")), span.get("text", ""))


def index_records(review_data):
    return {record_key(record): record for record in review_data.get("records", [])}


def label_map(record):
    return {label.get("label", ""): label for label in record.get("labels", [])}


def reviewed_span_set(review_data):
    """抽取 reviewer 最终认为 valid 的所有 span。"""
    spans = set()
    # 获得每个谓词被reviewed之后的所有label-span信息
    for record in review_data.get("records", []):
        for label in record.get("labels", []):
            for span in label.get("reviewed_spans", []):
                spans.add(span_key(record, label, span)) # 获得span_start, span_end, span_text的形式
    return spans


def candidate_span_set(review_data):
    """抽取原候选池中的所有 span。"""
    spans = set()
    for record in review_data.get("records", []):
        for label in record.get("labels", []):
            for span in label.get("candidate_spans", []):
                spans.add(span_key(record, label, span))
    return spans


def optional_map(review_data): # 看是否optional
    opts = {}
    for record in review_data.get("records", []):
        for label in record.get("labels", []):
            opts[label_key(record, label)] = bool(label.get("optional", False))
    return opts


def compute_agreement(review_a, review_b):
    """
    span-level agreement:
    Jaccard = |A ∩ B| / |A ∪ B|

    这里不把 optional 纳入主一致率；optional 只单独统计分歧数量。
    """


    spans_a = reviewed_span_set(review_a)
    spans_b = reviewed_span_set(review_b)
    # 每个元素是：('bn', "Speaking at the Cole 's home base in Norfolk , Virginia , Admiral John Foley said that the Navy is investigating the incident .", 'investigating', '21', 'ARG1', '21', '22', 'the incident')
    inter = spans_a & spans_b
    union = spans_a | spans_b

    diff = union - inter
    print(f'不一致的结果： {diff}')
    # import pdb;pdb.set_trace()
    # 每个元素组成是：('tc', 'oh they want their share .', 'want', '3', 'ARG1'): False} domain: sen, prd, prd_idx, label: optional
    opts_a = optional_map(review_a)
    opts_b = optional_map(review_b)
    all_opt_keys = set(opts_a) | set(opts_b)
    optional_disagreements = [
        key for key in all_opt_keys if opts_a.get(key, False) != opts_b.get(key, False)
    ]

    return {
        "reviewer_a_spans": len(spans_a),
        "reviewer_b_spans": len(spans_b),
        "span_intersection": len(inter),
        "span_union": len(union),
        "span_jaccard_agreement": len(inter) / len(union) if union else 1.0,
        "only_a": len(spans_a - spans_b),
        "only_b": len(spans_b - spans_a),
        "optional_disagreement_count": len(optional_disagreements),
    }


def build_disagreement_file(review_a, review_b):
    """
    导出需要仲裁的内容。
    - 两个人都标的 span: agreement
    - 只有 A 标的 span: only_a
    - 只有 B 标的 span: only_b

    你可以人工看 only_a / only_b，确认合理的加入 final reviewed set。
    """
    records_a = index_records(review_a)
    records_b = index_records(review_b)
    all_record_keys = sorted(set(records_a) | set(records_b), key=str)

    output = {
        "instructions": (
            "Review spans with status=only_a or only_b. If a span is valid, "
            "include it in the adjudicated final reviewed_spans before computing recall."
        ),
        "items": [],
    }

    for key in all_record_keys:
        rec_a = records_a.get(key)
        rec_b = records_b.get(key)
        rec = rec_a or rec_b
        labels_a = label_map(rec_a) if rec_a else {}
        labels_b = label_map(rec_b) if rec_b else {}
        all_labels = sorted(set(labels_a) | set(labels_b))

        for label_name in all_labels:
            label_a = labels_a.get(label_name, {})
            label_b = labels_b.get(label_name, {})
            spans_a = {
                simple_span_key(span): span for span in label_a.get("reviewed_spans", [])
            }
            spans_b = {
                simple_span_key(span): span for span in label_b.get("reviewed_spans", [])
            }
            candidate_keys = {
                simple_span_key(span)
                for span in (
                    label_a.get("candidate_spans", []) or label_b.get("candidate_spans", [])
                )
            }
            all_spans = sorted(set(spans_a) | set(spans_b), key=str)
            if not all_spans and bool(label_a.get("optional", False)) == bool(label_b.get("optional", False)):
                continue

            span_items = []
            for skey in all_spans:
                in_a = skey in spans_a
                in_b = skey in spans_b
                if in_a and in_b:
                    status = "agreement"
                    span = spans_a[skey]
                elif in_a:
                    status = "only_a"
                    span = spans_a[skey]
                else:
                    status = "only_b"
                    span = spans_b[skey]

                span_items.append(
                    {
                        "status": status,
                        "covered_by_candidate_pool": skey in candidate_keys,
                        "span": span,
                    }
                )

            optional_a = bool(label_a.get("optional", False))
            optional_b = bool(label_b.get("optional", False))
            if span_items or optional_a != optional_b:
                output["items"].append(
                    {
                        "domain": rec.get("domain", ""),
                        "sentence": rec.get("sentence", ""),
                        "prd_word": rec.get("prd_word", ""),
                        "prd_idx": rec.get("prd_idx", ""),
                        "label": label_name,
                        "role_description": label_a.get("role_description")
                        or label_b.get("role_description"),
                        "optional_a": optional_a,
                        "optional_b": optional_b,
                        "spans": span_items,
                    }
                )

    return output


def compute_candidate_recall(final_review):
    """
    span-level candidate recall:
    分母 = 仲裁后全部 valid spans
    分子 = 这些 valid spans 中原本已在 candidate_spans 里的 span
    """
    total = 0
    covered = 0
    human_added_not_covered = 0
    per_domain = defaultdict(lambda: {"total": 0, "covered": 0, "human_added_not_covered": 0})

    for record in final_review.get("records", []):
        domain = record.get("domain", "unknown")
        for label in record.get("labels", []):
            candidate_keys = {simple_span_key(span) for span in label.get("candidate_spans", [])}
            for span in label.get("reviewed_spans", []):
                total += 1
                per_domain[domain]["total"] += 1

                is_covered = simple_span_key(span) in candidate_keys
                if is_covered:
                    covered += 1
                    per_domain[domain]["covered"] += 1
                elif span.get("source") == "human_added":
                    human_added_not_covered += 1
                    per_domain[domain]["human_added_not_covered"] += 1

    return {
        "covered": covered,
        "total": total,
        "recall": covered / total if total else 0.0,
        "human_added_not_covered": human_added_not_covered,
        "per_domain": {
            domain: {
                **stats,
                "recall": stats["covered"] / stats["total"] if stats["total"] else 0.0,
            }
            for domain, stats in sorted(per_domain.items())
        },
    }


def print_agreement(stats):
    print("=" * 72)
    print("Two-reviewer span-level agreement")
    print("=" * 72)
    print(f"Reviewer A valid spans: {stats['reviewer_a_spans']}")
    print(f"Reviewer B valid spans: {stats['reviewer_b_spans']}")
    print(f"Intersection: {stats['span_intersection']}")
    print(f"Union: {stats['span_union']}")
    print(f"Span-level Jaccard agreement: {stats['span_jaccard_agreement'] * 100:.2f}%")
    print(f"Only A: {stats['only_a']}")
    print(f"Only B: {stats['only_b']}")
    print(f"Optional disagreements: {stats['optional_disagreement_count']}")
    print("=" * 72)


def print_recall(stats):
    print("=" * 72)
    print("Candidate pool span-level recall")
    print("=" * 72)
    print(f"Overall: {stats['covered']} / {stats['total']} = {stats['recall'] * 100:.2f}%")
    print(f"Human-added valid spans not covered: {stats['human_added_not_covered']}")
    print("-" * 72)
    for domain, domain_stats in stats["per_domain"].items():
        print(
            f"{domain}: {domain_stats['covered']} / {domain_stats['total']} "
            f"= {domain_stats['recall'] * 100:.2f}% "
            f"(human-added not covered: {domain_stats['human_added_not_covered']})"
        )
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two span-recall review files and compute span-level recall."
    )
    parser.add_argument("--reviewer-a", default=DEFAULT_REVIEWER_A)
    parser.add_argument("--reviewer-b", default=DEFAULT_REVIEWER_B)
    parser.add_argument("--out-disagreements", default=DEFAULT_DISAGREEMENTS)
    parser.add_argument(
        "--final-review",
        default=None,
        help="Adjudicated final review JSON. If provided, compute candidate span-level recall.",
    )
    args = parser.parse_args()

    # 读取两个审核者的文件, ['settings', 'records', 'last_index', 'updated_at']
    review_a = read_json(args.reviewer_a)
    review_b = read_json(args.reviewer_b)

    agreement = compute_agreement(review_a, review_b)
    print_agreement(agreement)
    # import pdb;pdb.set_trace()
    # disagreements = build_disagreement_file(review_a, review_b)
    # write_json(args.out_disagreements, disagreements)
    # print(f"Disagreement file written to: {args.out_disagreements}")
    # print(f"Items requiring inspection: {len(disagreements['items'])}")

    if args.final_review:
        recall = compute_candidate_recall(read_json(args.final_review))
        print_recall(recall)


if __name__ == "__main__":
    main()
