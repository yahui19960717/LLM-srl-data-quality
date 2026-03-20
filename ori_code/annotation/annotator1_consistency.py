"""
标注者的标注准确率
比较 final 文件（审核完成的，我和王老师，一个用审核界面，一个直接在原始标注的数据来标注）和标注者文件中的 selected_spans
规则：只要标注者的 selected_spans 中有至少一个 span 与 gold 文件中的任意一个 span 完全匹配，则视为正确。

"""

import json
import argparse
from pathlib import Path


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def span_to_tuple(span: dict) -> tuple:
    """将 span 转换为可比较的元组 (start, end, text)"""
    return (span["start"], span["end"], span["text"])


def compare_spans(gold_file: str, annotator_file: str):
    gold_data = read_json(gold_file)
    ann_data = read_json(annotator_file)

    gold_annotations = gold_data.get("annotations", [])
    ann_annotations = ann_data.get("annotations", [])

    # 用 idx 建立索引
    gold_map = {item["idx"]: item for item in gold_annotations}
    ann_map = {item["idx"]: item for item in ann_annotations}

    all_ids = sorted(set(gold_map.keys()) | set(ann_map.keys()))

    total = 0
    correct = 0
    results = []

    for idx in all_ids:
        gold_item = gold_map.get(idx)
        ann_item = ann_map.get(idx)

        if gold_item is None:
            results.append({
                "idx": idx,
                "status": "MISSING_IN_GOLD",
                "sentence": ann_item.get("sentence", ""),
                "note": "该条目在 gold 文件中不存在"
            })
            continue

        if ann_item is None:
            results.append({
                "idx": idx,
                "status": "MISSING_IN_ANNOTATOR",
                "sentence": gold_item.get("sentence", ""),
                "note": "该条目在标注者文件中不存在"
            })
            total += 1
            continue

        total += 1
        sentence = gold_item.get("sentence", "")
        prd_word = gold_item.get("prd_word", "")
        label = gold_item.get("label", "")

        gold_spans = set(span_to_tuple(s) for s in gold_item.get("selected_spans", []))
        ann_spans = set(span_to_tuple(s) for s in ann_item.get("selected_spans", []))

        # 只要有一个标注者的 span 命中 gold 中任意一个 span，即为正确
        matched = gold_spans & ann_spans
        is_correct = len(matched) > 0

        if is_correct:
            correct += 1

        results.append({
            "idx": idx,
            "status": "CORRECT" if is_correct else "WRONG",
            "sentence": sentence,
            "prd_word": prd_word,
            "label": label,
            "gold_spans": list(gold_item.get("selected_spans", [])),
            "ann_spans": list(ann_item.get("selected_spans", [])),
            "matched_spans": [{"start": s[0], "end": s[1], "text": s[2]} for s in matched]
        })

    # 打印结果
    print("=" * 70)
    print(f"{'比较结果汇总':^70}")
    print("=" * 70)
    print(f"总条目数（有效对比）: {total}")
    print(f"正确数: {correct}")
    print(f"错误数: {total - correct}")
    print(f"准确率: {correct / total * 100:.2f}%" if total > 0 else "准确率: N/A")
    print("=" * 70)
    return correct, total
    # print("\n详细结果：\n")
    # for r in results:
    #     status_icon = "✅" if r["status"] == "CORRECT" else ("❌" if r["status"] == "WRONG" else "⚠️")
    #     print(f"{status_icon} [idx={r['idx']}] {r['status']}")
    #     print(f"   句子: {r.get('sentence', '')}")
    #     if r["status"] in ("CORRECT", "WRONG"):
    #         print(f"   谓词: {r.get('prd_word', '')}  |  标签: {r.get('label', '')}")
    #         print(f"   Gold spans:      {[s['text'] for s in r.get('gold_spans', [])]}")
    #         print(f"   标注者 spans:    {[s['text'] for s in r.get('ann_spans', [])]}")
    #         if r["status"] == "CORRECT":
    #             print(f"   命中 spans:      {[s['text'] for s in r.get('matched_spans', [])]}")
    #     else:
    #         print(f"   备注: {r.get('note', '')}")
    #     print()

    # # 保存结果到 JSON
    # output_path = Path(gold_file).parent / "comparison_result.json"
    # summary = {
    #     "total": total,
    #     "correct": correct,
    #     "wrong": total - correct,
    #     "accuracy": round(correct / total * 100, 2) if total > 0 else None,
    #     "details": results
    # }
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(summary, f, ensure_ascii=False, indent=2)
    # print(f"详细结果已保存至: {output_path}")


if __name__ == "__main__":

    # single gold random
    final_file = "annotated_final/annotations_single_gold_random_wlj_final.json"
    annotator_wlj = "anno/annotations_single_gold_random30_wlj.json" #96.67%
    annotator_lyh = "anno/annotations_single_gold_random30_lyh.json" #70%
    correct1, total1 =compare_spans(final_file, annotator_wlj)
    correct2, total2 =compare_spans(final_file, annotator_lyh)

    # # # small model 163
    final_file = "annotated_final/annotations_wlj_smallmodel_163_final.json"
    annotator_wlj = "anno/annotations_wlj_smallmodel_163.json" # 84.05%
    annotator_lyh = "anno/annotations_lyh_smallmodel_163.json" # 73.62%
    correct11, total11 =compare_spans(final_file, annotator_wlj)
    correct22, total22 =compare_spans(final_file, annotator_lyh)



    final_file = "annotated_final/annotations_single_gold_wlj_final.json"
    annotator_wlj = "anno/annotations_single_gold_wlj.json" # 73.45%
    annotator_lyh = "anno/annotations_single_gold_lyh.json" # 62.83%
    correct111, total111 =compare_spans(final_file, annotator_wlj)
    correct222, total222 =compare_spans(final_file, annotator_lyh)

    final_file = "annotated_final/annotations_gold_o1right_random_wlj_final.json"
    annotator_wlj = "anno/annotations_gold_o1right_random_wlj.json" # 93.33%
    annotator_lyh = "anno/annotations_gold_o1right_random_lyh.json" # 96.67%
    correct1111, total1111 =compare_spans(final_file, annotator_wlj)
    correct2222, total2222 =compare_spans(final_file, annotator_lyh)

    final_file = "annotated_final/annotations_smallmodel_botherror_random_wlj_final.json"
    annotator_wlj = "anno/annotations_smallmodel_botherror_random_wlj.json" # 40%
    annotator_lyh = "anno/annotations_smallmodel_botherror_random_lyh.json" # 36.67%
    correct11111, total11111 =compare_spans(final_file, annotator_wlj)
    correct22222, total22222 =compare_spans(final_file, annotator_lyh)
    # all_correct1 = correct1+ correct11+correct111+correct1111+correct11111
    # all_correct2 = correct2+ correct22+correct222+correct2222+correct22222


    final_file = "annotated_final/annotations_single_gold_o1wrongdsright_93_wlj_final.json"
    annotator_wlj = "anno/annotations_single_gold_o1wrongdsright_93_wlj.json" #  86.02
    annotator_lyh = "anno/annotations_single_gold_o1wrongdsright_93_lyh.json" #  67.74
    correct111111, total111111 =compare_spans(final_file, annotator_wlj)
    correct222222, total222222 =compare_spans(final_file, annotator_lyh)
    all_correct1 = correct1+ correct11+correct111+correct1111+correct11111+correct111111
    all_correct2 = correct2+ correct22+correct222+correct2222+correct22222+correct222222

    totalall = total1+total11+total111+total1111+total11111+total111111

    assert totalall == total2+total22+total222+total2222+total22222+total222222
    print(correct1+ correct11+correct111+correct1111+correct11111)
    print(correct2+ correct22+correct222+correct2222+correct22222)
    print(total1+total11+total111+total1111+total11111)
    print(f'{all_correct1} / {totalall} = {all_correct1/totalall:.2%}')
    print(f'{all_correct2} / {totalall} = {all_correct2/totalall:.2%}')
    # compare_spans(final_file, annotator_wlj)
    # compare_spans(final_file, annotator_lyh)

    # 77%;67.958
