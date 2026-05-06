import argparse
import json
from pathlib import Path


# 默认要比较的标注者文件对。
# 每一项格式为：
# (结果名称, 标注者A文件路径, 标注者B文件路径)
# 如果运行脚本时不传 --pair，就会自动计算这些文件对的一致率。
DEFAULT_PAIRS = [
    # # old
    # (
    #     "single_gold_random30",
    #     "anno/annotations_single_gold_random30_wlj.json",
    #     "anno/annotations_single_gold_random30_lyh.json",
    # ),
    # (
    #     "smallmodel_163",
    #     "anno/annotations_wlj_smallmodel_163.json",
    #     "anno/annotations_lyh_smallmodel_163.json",
    # ),
    # (
    #     "single_gold",
    #     "anno/annotations_single_gold_wlj.json",
    #     "anno/annotations_single_gold_lyh.json",
    # ),
    # (
    #     "gold_o1right_random",
    #     "anno/annotations_gold_o1right_random_wlj.json",
    #     "anno/annotations_gold_o1right_random_lyh.json",
    # ),
    # (
    #     "smallmodel_botherror_random",
    #     "anno/annotations_smallmodel_botherror_random_wlj.json",
    #     "anno/annotations_smallmodel_botherror_random_lyh.json",
    # ),
    # (
    #     "single_gold_o1wrongdsright_93",
    #     "anno/annotations_single_gold_o1wrongdsright_93_wlj.json",
    #     "anno/annotations_single_gold_o1wrongdsright_93_lyh.json",
    # ),
    # bn  有optional的文件
    (
        "sm_recallright_110",
        "anno/annotation_smnotrecallright_110_wlj.json",
        "anno/annotation_smnotrecallright_110_lyh.json",
    ),   
    (
        "sm_overlap",
        "anno/annotation_smallmodel_overlap_wlj.json",
        "anno/annotation_smallmodel_overlap_lyh.json",
    ),
    
    (
        "sm_recall21",
        "anno/annotation_smnotrecall_21_wlj.json",
        "anno/annotation_smnotrecall_21_lyh.json",
    ),
    (
        "sm_recallright_random30",
        "anno/annotation_smnotrecallright_random30_wlj.json",
        "anno/annotation_smnotrecallright_random30_lyh.json",
    ),
    # tc
    (
        "tc_smallmodel",
        "anno/tc/annotation_tc_smallmodel_removerepeate_322_wlj.json",
        "anno/tc/annotation_tc_smallmodel_removerepeate_322_lyh.json",
    ),
    (
        "tc_single_gold_201",
        "anno/tc/annotation_tc_single_gold_201_wlj.json",
        "anno/tc/annotation_tc_single_gold_201_lyh.json",
    ),
    (
        "tc_o1wrong_dsright_80",
        "anno/tc/annotation_tc_o1wrong_dsright_80_wlj.json",
        "anno/tc/annotation_tc_o1wrong_dsright_80_lyh.json",
    ),
]


def read_json(path):
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_annotations(data):
    """
    兼容两种数据格式：
    1. {"annotations": [...]} 这种标注工具导出的格式；
    2. [...] 这种直接存 list 的格式。
    最终统一返回 annotations 列表。
    """
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be a list or a dict with an 'annotations' field.")


def item_id(item):
    """
    为每条标注构造唯一 id，用来对齐两个标注者的同一条样本。
    优先使用 idx；如果没有 idx，就退回到
    (sentence, predicate word, predicate index, label)。
    """
    if "idx" in item:
        return item["idx"]
    return (
        item.get("sentence", item.get("sen", "")),
        item.get("prd_word", ""),
        item.get("prd_idx", item.get("pred.idx", "")),
        item.get("label", ""),
    )


def span_tuple(span, include_text=False):
    """
    将 span 转成可比较的 tuple。
    默认只比较 start/end；如果加 --include-text，则 text 也必须一致。
    """
    if include_text:
        return (span.get("start"), span.get("end"), span.get("text", ""))
    return (span.get("start"), span.get("end"))


def span_set(item, include_text=False):
    """取出某条样本中所有 selected_spans，并转成 set，便于比较。"""
    return {span_tuple(span, include_text) for span in item.get("selected_spans", [])}


def is_optional(item):
    """
    判断一条样本是否为 optional annotation。
    兼容 bool 和常见字符串写法。
    """
    value = item.get("optional", False)
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1", "optional", "可标可不标"}
    return bool(value)


def compare_item(item_a, item_b, include_text=False):
    """
    比较两个标注者在同一条样本上的一致性。

    返回两种一致率：
    1. exact / strict：
       两个标注者选择的 span 集合必须完全相同。
    2. relaxed：
       和论文的新 evaluation protocol 对齐：
       - 如果两个 span 集合完全相同，算一致；
       - 如果双方都标注了optional为True，算一致
       - 如果双方optional标注不一致，则继续比较span；
       - 如果存在多个有效 span，只要两个标注者有任意一个 span 重合，算一致；
       - 否则算不一致。
    """
    # import pdb;pdb.set_trace()
    spans_a = span_set(item_a, include_text=include_text)
    spans_b = span_set(item_b, include_text=include_text)
    optional_a = is_optional(item_a)
    optional_b = is_optional(item_b)
    
    both_optional = optional_a and optional_b

    exact = spans_a == spans_b

    if both_optional:
    # Optional annotation agreement:
        # if both annotators mark the same item as optional, count it as
        # agreement. If only one side marks optional, do not decide here;
        # continue to compare selected spans below.
        relaxed = True
        reason = "both_optional"
    elif exact:
        relaxed = True
        reason = "exact"
        
    elif spans_a and spans_b and (spans_a & spans_b):
        # Multiple valid spans: if the annotators share at least one selected
        # span, they agree under the relaxed protocol.
        relaxed = True
        reason = "span_overlap"
    else:
        relaxed = False
        reason = "mismatch"

    return {
        "exact": exact,
        "relaxed": relaxed,
        "reason": reason,
        "spans_a": sorted(spans_a),
        "spans_b": sorted(spans_b),
        "optional_a": optional_a,
        "optional_b": optional_b,
    }


def compare_files(name, file_a, file_b, include_text=False, details_dir=None):
    """
    计算一对标注者文件的一致率。

    主要步骤：
    1. 读取两个 JSON 文件；
    2. 用 item_id 对齐两个标注者的样本；
    3. 对每条样本计算 strict 和 relaxed agreement；
    4. 汇总当前文件对的一致率；
    5. 如果指定 details_dir，则把 relaxed 不一致的样本保存出来方便检查。
    """
    data_a = get_annotations(read_json(file_a))
    data_b = get_annotations(read_json(file_b))

    map_a = {item_id(item): item for item in data_a} # map_a 是标注者A的样本id到样本的映射
    map_b = {item_id(item): item for item in data_b} # map_b 是标注者B的样本id到样本的映射
    all_ids = sorted(set(map_a) | set(map_b), key=str)
    # import pdb;pdb.set_trace()
    total = 0
    exact_agree = 0
    relaxed_agree = 0
    missing_a = 0
    missing_b = 0
    reasons = {
        "exact": 0,
        "both_optional": 0,
        "span_overlap": 0,
        "mismatch": 0,
        "missing_in_a": 0,
        "missing_in_b": 0,
    }
    details = []

    for key in all_ids:
        item_a = map_a.get(key) # 获得标注者A的样本
        item_b = map_b.get(key) # 获得标注者B的样本

        if item_a is None: # 如果标注者A的样本不存在，则认为标注者A缺失
            missing_a += 1
            reasons["missing_in_a"] += 1
            details.append({"id": key, "status": "missing_in_a"})
            continue
        if item_b is None:
            missing_b += 1
            reasons["missing_in_b"] += 1
            details.append({"id": key, "status": "missing_in_b"})
            continue

        total += 1
        result = compare_item(item_a, item_b, include_text=include_text)
        exact_agree += int(result["exact"])
        relaxed_agree += int(result["relaxed"])
        reasons[result["reason"]] += 1

        if not result["relaxed"]:
            details.append(
                {
                    "id": key,
                    "status": "mismatch",
                    "sentence": item_a.get("sentence", item_a.get("sen", "")),
                    "prd_word": item_a.get("prd_word", ""),
                    "label": item_a.get("label", ""),
                    "spans_a": result["spans_a"],
                    "spans_b": result["spans_b"],
                    "optional_a": result["optional_a"],
                    "optional_b": result["optional_b"],
                }
            )

    exact_rate = exact_agree / total if total else 0.0
    relaxed_rate = relaxed_agree / total if total else 0.0

    summary = {
        "name": name,
        "file_a": str(file_a),
        "file_b": str(file_b),
        "total_compared": total,
        "exact_agree": exact_agree,
        "exact_rate": exact_rate,
        "relaxed_agree": relaxed_agree,
        "relaxed_rate": relaxed_rate,
        "missing_in_a": missing_a,
        "missing_in_b": missing_b,
        "reasons": reasons,
    }

    if details_dir:
        details_dir.mkdir(parents=True, exist_ok=True)
        out = details_dir / f"{name}_mismatches.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        summary["details_file"] = str(out)

    return summary


def parse_pair(spec):
    """解析命令行传入的 name:file_a:file_b 格式。"""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Each --pair must use the format name:annotator_a.json:annotator_b.json"
        )
    name, file_a, file_b = parts
    return name, Path(file_a), Path(file_b)


def main():
    """
    脚本入口。

    默认行为：
    - 不传参数时，计算 DEFAULT_PAIRS 中所有文件对；
    - 输出每个文件对的一致率；
    - 输出所有文件的 micro / macro 平均一致率。

    可选行为：
    - 用 --pair 临时指定其他文件对；
    - 用 --include-text 要求 span text 也一致；
    - 用 --details-dir 保存不一致样本。
    """
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement for SRL span annotations."
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="File pair in the format name:annotator_a.json:annotator_b.json. Can be repeated.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Require span text to match in addition to start/end offsets.",
    )
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=None,
        help="Optional directory for mismatch JSON files.",
    )
    args = parser.parse_args()

    pair_specs = args.pair
    if pair_specs is None:
        pairs = [(name, Path(file_a), Path(file_b)) for name, file_a, file_b in DEFAULT_PAIRS]
    else:
        pairs = [parse_pair(spec) for spec in pair_specs]

    summaries = []
    for name, file_a, file_b in pairs:
        summaries.append(
            compare_files(
                name=name,
                file_a=file_a,
                file_b=file_b,
                include_text=args.include_text,
                details_dir=args.details_dir,
            )
        )

    total = sum(s["total_compared"] for s in summaries)
    # for s in summaries:
    #     print(s['total_compared'])
    # import pdb;pdb.set_trace()
    micro_exact = sum(s["exact_agree"] for s in summaries) / total if total else 0.0
    micro_relaxed = sum(s["relaxed_agree"] for s in summaries) / total if total else 0.0 # 所有文件一致的样本总数/所有文件可比较样本数
    macro_exact = sum(s["exact_rate"] for s in summaries) / len(summaries)
    macro_relaxed = sum(s["relaxed_rate"] for s in summaries) / len(summaries) # 每个文件计算，然后算平均
    # print(sum(s["missing_in_a"] for s in summaries) / len(summaries))
    # print(sum(s["missing_in_b"] for s in summaries) / len(summaries))

    print("=" * 96)
    print(
        f"{'Name':<34} {'N':>6} {'Strict':>10} {'Relaxed':>10} "
        f"{'Exact':>7} {'Overlap':>8} {'Both_Optional':>9} {'Mismatch':>9}"
    )
    print("-" * 96)
    for s in summaries:
        reasons = s["reasons"]
        print(
            f"{s['name']:<34} {s['total_compared']:>6} "
            f"{s['exact_rate'] * 100:>9.2f}% {s['relaxed_rate'] * 100:>9.2f}% "
            f"{reasons['exact']:>7} {reasons['span_overlap']:>8} "
            f"{reasons['both_optional']:>9} {reasons['mismatch']:>9}"
        )

    print("-" * 96)
    print(f"Micro average strict agreement:  {micro_exact * 100:.2f}%")
    print(f"Micro average relaxed agreement: {micro_relaxed * 100:.2f}%")
    print(f"Macro average strict agreement:  {macro_exact * 100:.2f}%")
    print(f"Macro average relaxed agreement: {macro_relaxed * 100:.2f}%")
    print("=" * 96)


if __name__ == "__main__":
    main()
