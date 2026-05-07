# 首先抽取50个谓词，每个文件（bn+tc)各25个，
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


# 标注的数据路径
DEFAULT_BN_FILE = "analysis/test_bn_500_core_final_v4.json"
DEFAULT_TC_FILE = "analysis/test_tc_core_final_v2.json"
CORE_LABELS = [f"ARG{i}" for i in range(6)]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_grammar_error(item): # 句子是否有语法
    """过滤明显语法错误样本。"""
    status = item.get("grammar_status", "")
    desc = item.get("grammar_error_desc", "")
    return status == "有语法错误" or desc == "有语法错误"


def parse_key(key): # 解析的数据
    """final json 的 key 格式：sentence\tpredicate\tpredicate_index\tlabel。"""
    sen, prd_word, prd_idx, label = key.split("\t")
    return sen, prd_word, int(prd_idx), label


def group_by_predicate(data, keep_all_core_labels=False):
    """
    将 label 级别数据聚合到谓词级别。

    输入：
    {
      "sentence\tprd_word\tprd_idx\tARG0": {...},
      "sentence\tprd_word\tprd_idx\tARG1": {...}
    }

    输出：
    {
      (sentence, prd_word, prd_idx): {
        "sentence": ...,
        "prd_word": ...,
        "prd_idx": ...,
        "labels": {
          "ARG0": item,
          "ARG1": item
        }
      }
    }

    如果 keep_all_core_labels=True，会给 ARG0-ARG5 中缺失的 label
    补一个空记录，便于人工检查“该 role 是否真的不存在”。
    """
    grouped = {}

    for key, item in data.items():
        sen, prd_word, prd_idx, label = parse_key(key)
        if label not in CORE_LABELS:
            continue
        if is_grammar_error(item):
            continue

        pred_key = (sen, prd_word, prd_idx)
        if pred_key not in grouped:
            grouped[pred_key] = {
                "sentence": sen,
                "prd_word": prd_word,
                "prd_idx": prd_idx,
                "labels": {},
            }
        grouped[pred_key]["labels"][label] = item

    if keep_all_core_labels:
        for pred in grouped.values():
            for label in CORE_LABELS:
                pred["labels"].setdefault(
                    label,
                    {
                        "label": label,
                        "optional": False,
                        "selected_spans": [],
                        "span_mean": None,
                        "type": "missing_label_record",
                    },
                )

    return grouped


def item_to_span_rows(domain, pred_id, pred, label, item):
    """
    将一个 label 的 selected_spans 展开成输出行。
    如果 selected_spans 为空，也输出一行，span 字段留空。
    """
    optional = bool(item.get("optional", False))
    spans = item.get("selected_spans", [])

    base = {
        "domain": domain,
        "predicate_id": pred_id,
        "sentence": pred["sentence"],
        "prd_word": pred["prd_word"],
        "prd_idx": pred["prd_idx"],
        "label": label,
        "optional": optional,
        "span_mean": item.get("span_mean"),
        "source_type": item.get("type"),
    }

    if not spans:
        return [
            {
                **base,
                "has_span": False,
                "span_start": "",
                "span_end": "",
                "span_text": "",
                "span_models": "",
            }
        ]

    rows = []
    for span in spans:
        rows.append(
            {
                **base,
                "has_span": True,
                "span_start": span.get("start", ""),
                "span_end": span.get("end", ""),
                "span_text": span.get("text", ""),
                "span_models": "|".join(span.get("models", [])),
            }
        )
    return rows


def sample_predicates(grouped, sample_size, seed):
    """从谓词级别数据中随机抽样。"""
    rng = random.Random(seed)
    keys = sorted(grouped.keys(), key=str)
    if sample_size > len(keys):
        raise ValueError(f"sample_size={sample_size} exceeds predicate count={len(keys)}")
    sampled_keys = rng.sample(keys, sample_size)
    return [(key, grouped[key]) for key in sampled_keys]


def build_outputs(domain, sampled_predicates):
    """
    构造两种输出：
    1. nested JSON：按谓词组织，方便人工查看；
    2. flat rows：每个 span 一行，方便导入表格统计。
    """
    nested = []
    rows = []

    for i, (pred_key, pred) in enumerate(sampled_predicates, start=1):
        pred_id = f"{domain}_{i:03d}"
        labels = []

        for label in sorted(pred["labels"]):
            item = pred["labels"][label]
            label_record = {
                "label": label,
                "optional": bool(item.get("optional", False)),
                "span_mean": item.get("span_mean"),
                "source_type": item.get("type"),
                "spans": [
                    {
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "text": span.get("text", ""),
                        "models": span.get("models", []),
                    }
                    for span in item.get("selected_spans", [])
                ],
            }
            labels.append(label_record)
            rows.extend(item_to_span_rows(domain, pred_id, pred, label, item))

        nested.append(
            {
                "predicate_id": pred_id,
                "sentence": pred["sentence"],
                "prd_word": pred["prd_word"],
                "prd_idx": pred["prd_idx"],
                "labels": labels,
            }
        )

    return nested, rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "predicate_id",
        "sentence",
        "prd_word",
        "prd_idx",
        "label",
        "optional",
        "span_mean",
        "source_type",
        "has_span",
        "span_start",
        "span_end",
        "span_text",
        "span_models",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Sample predicates and export all label/span records for recall evaluation."
    )
    parser.add_argument("--bn-file", default=DEFAULT_BN_FILE)
    parser.add_argument("--tc-file", default=DEFAULT_TC_FILE)
    parser.add_argument("--sample-size", type=int, default=25, help="Predicates per domain.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--all-core-labels",
        action="store_true",
        help="Also output ARG0-ARG5 labels absent from the final json as empty records.",
    )
    parser.add_argument("--out-json", default="analysis/sample_50_predicate_arguments.json")
    parser.add_argument("--out-csv", default="analysis/sample_50_predicate_arguments.csv")
    args = parser.parse_args()

    # 以谓词分组，获得每个谓词最终的结果，去掉语法错误的句子
    # 1238个谓词
    bn = group_by_predicate(read_json(args.bn_file), keep_all_core_labels=args.all_core_labels)
    # 1149个谓词
    tc = group_by_predicate(read_json(args.tc_file), keep_all_core_labels=args.all_core_labels)
    # import pdb;pdb.set_trace()

    # 每个文件采样 25个
    sampled_bn = sample_predicates(bn, args.sample_size, args.seed)
    sampled_tc = sample_predicates(tc, args.sample_size, args.seed + 1)

    # 输出结果
    nested_bn, rows_bn = build_outputs("bn", sampled_bn)
    nested_tc, rows_tc = build_outputs("tc", sampled_tc)

    output = {
        "settings": {
            "bn_file": args.bn_file,
            "tc_file": args.tc_file,
            "sample_size_per_domain": args.sample_size,
            "seed_bn": args.seed,
            "seed_tc": args.seed + 1,
            "grammar_errors_filtered": True,
            "all_core_labels": args.all_core_labels,
        },
        "bn": nested_bn,
        "tc": nested_tc,
    }
    rows = rows_bn + rows_tc

    write_json(Path(args.out_json), output)
    write_csv(Path(args.out_csv), rows)

    print(f"BN predicates after filtering: {len(bn)}")
    print(f"TC predicates after filtering: {len(tc)}")
    print(f"Sampled predicates: {len(sampled_bn) + len(sampled_tc)}")
    print(f"Span/label rows: {len(rows)}")
    print(f"JSON written to: {args.out_json}")
    print(f"CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()

# from collections import defaultdict
# from config import *
# import random 
# random.seed(1)
# bn = read_json(f"analysis/test_bn_500_core_final_v4.json")
# tc = read_json(f"analysis/test_tc_core_final_v2.json")
# new_bn, new_tc = defaultdict(list), defaultdict(list)
# for key in bn.keys():
#     sen, prd_word, prd_idx, label = key.split("\t")
#     if "\t".join([sen, prd_word, prd_idx]) not in new_bn.keys():
#         new_bn["\t".join([sen, prd_word, prd_idx])] = []
#         new_bn["\t".join([sen, prd_word, prd_idx])].append(bn[key]) # 1240个谓词
#     else:
#         new_bn["\t".join([sen, prd_word, prd_idx])].append(bn[key]) # 1240个谓词
#     if len(new_bn["\t".join([sen, prd_word, prd_idx])]) >=2:
#         import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()
# for key in tc.keys():
#     sen, prd_word, prd_idx, label = key.split("\t")
#     if "\t".join([sen, prd_word, prd_idx]) not in new_tc.keys():
#         new_tc["\t".join([sen, prd_word, prd_idx])] = []
#         new_tc["\t".join([sen, prd_word, prd_idx])].append(tc[key]) #1150个谓词
#     else:
#         new_tc["\t".join([sen, prd_word, prd_idx])].append(tc[key]) #1150个谓词



