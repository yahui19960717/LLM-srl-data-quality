#!/usr/bin/env python3
"""
相比于上一个版本，这个版本增加了可标可不标注的处理
抽取不匹配的标注数据，生成文本 供 Streamlit 查看器使用。
处理流程：
  1. 加载双方 JSON，去重
  2. 过滤掉任意一方标记为有语法错误的条目（grammar_status != "没有语法错误"）
  3. 对齐双方共同标注的 idx
  4. 计算匹配类型，抽取不一致条目
规则：只要标注者的 selected_spans 中有至少一个 span 与 gold 文件中的任意一个 span 完全匹配，则视为正确。
用法:
    python extract_disagreements.py \
        --file_1 wlj.json \
        --file_2 lyh.json \
        --output disagreements.json
"""

import json
import argparse
from typing import Dict, List, Tuple
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import os
def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

def remove_duplicates(data: List[Dict]) -> List[Dict]:
    seen, result = set(), []
    for item in data:
        if item['idx'] not in seen:
            seen.add(item['idx'])
            result.append(item)
    return result
    
def filter_grammar_errors(data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    过滤掉有语法错误的条目。
    grammar_status 字段值为 "没有语法错误" 时保留，其余全部剔除。
    字段缺失时视为无语法错误（保留）。

    返回：(clean_data, removed_items)
    """
    clean, removed = [], []
    for item in data:
        status = item.get("grammar_status", "没有语法错误").strip()
        if status == "没有语法错误":
            clean.append(item)
        else:
            removed.append(item)
            print(item)
    return clean, removed

def align_annotations(data_a: List[Dict], data_b: List[Dict]) -> List[Tuple[Dict, Dict]]:
    dict_a = {item['idx']: item for item in data_a}
    dict_b = {item['idx']: item for item in data_b}
    common = set(dict_a.keys()) & set(dict_b.keys())
    # dict_a = {(item['sentence'], item["prd_word"], item["prd_idx"], item['label']):item for item in data_a}
    # dict_b = {(item['sentence'], item["prd_word"], item["prd_idx"], item['label']):item for item in data_b}
    # common = set(dict_a.keys()) & set(dict_b.keys())
    for idx in sorted(common):
        assert dict_a[idx]['sentence'] == dict_b[idx]['sentence']
        assert dict_a[idx]['prd_word'] == dict_b[idx]['prd_word']
        assert dict_a[idx]['prd_idx'] == dict_b[idx]['prd_idx']
        assert dict_a[idx]['label'] == dict_b[idx]['label']
    aligned = [(dict_a[idx], dict_b[idx]) for idx in sorted(common)]
    return aligned


def count_exact_span_pairs(spans_a: List[Dict], spans_b: List[Dict]) -> int:
    used_b = [False] * len(spans_b) # 标记b是否匹配
    count = 0
    for sa in spans_a: # 遍历a
        for j, sb in enumerate(spans_b):
            if not used_b[j] and sa['start'] == sb['start'] and sa['end'] == sb['end']: # 开始和结束匹配就是匹配
                count += 1
                used_b[j] = True
                break
    return count

def classify_match(spans_a: List[Dict], spans_b: List[Dict]) -> Tuple[str, float]:

    # 看下是否没有论元
    if not spans_a and not spans_b:
        return "both_empty", 1.0
    if not spans_a:
        return "empty_a", 0.0
    if not spans_b:
        return "empty_b", 0.0

    exact_pairs = count_exact_span_pairs(spans_a, spans_b)
    max_len = max(len(spans_a), len(spans_b))
    score = round(exact_pairs / max_len, 3) # 单个标注的匹配得分（匹配/最大的长度）

    if len(spans_a) == len(spans_b) and exact_pairs == len(spans_a): # 全部匹配
        return "exact", 1.0
    if exact_pairs > 0: # 部分匹配
        return "partial", score
    return "none", 0.0  # 两个列表都有论元（都不为空），但完全没有一个 span 能完全匹配

DISAGREEMENT_TYPES = {"partial", "none", "empty_a", "empty_b"}

def extract(aligned, grammar_filter_stats, file_a, file_b, name_1, name_2 ):
    records = []
    stats = {
        "total_aligned": len(aligned),
        "exact": 0, "both_empty": 0,
        "partial": 0, "none": 0, "empty_a": 0, "empty_b": 0, "optional": 0, 
    }
    for item_a, item_b in aligned:
        # import pdb;pdb.set_trace()

        optional_a = item_a.get('optional', None)
        optional_b = item_b.get('optional', None)
        if optional_a == optional_b and optional_b == True:
            stats["optional"] = stats.get("optional", 0) + 1 
        spans_a = item_a.get('selected_spans', []) # 每个人标注的全部数据
        spans_b = item_b.get('selected_spans', [])

        match_type, score = classify_match(spans_a, spans_b) # 看下双方spans的匹配数
        stats[match_type] = stats.get(match_type, 0) + 1 #对应的匹配类型+1

        if match_type not in DISAGREEMENT_TYPES: # 如果精确匹配就不用管了
            continue

        # 记录 partial 时已配对的 span 索引
        matched_a_idx, matched_b_idx = set(), set() # 对于部分匹配的，得到匹配的结果
        if match_type == "partial":
            used_b = [False] * len(spans_b)
            for i, sa in enumerate(spans_a):
                for j, sb in enumerate(spans_b):
                    if not used_b[j] and sa['start'] == sb['start'] and sa['end'] == sb['end']:
                        matched_a_idx.add(i)
                        matched_b_idx.add(j)
                        used_b[j] = True
                        break

        records.append({
            "idx":               item_a['idx'],
            "sentence":          item_a['sentence'],
            "prd_word":          item_a.get('prd_word', ''),
            "prd_idx":           item_a.get('prd_idx', None),
            "label":             item_a['label'],
            "span_mean":         item_a.get('span_mean', ''),
            "match_type":        match_type,
            "score":             score,
            "spans_a":           spans_a,
            "spans_b":           spans_b,
            "matched_a_idx":     list(matched_a_idx),
            "matched_b_idx":     list(matched_b_idx),
            "optional_a":        item_a.get('optional', False),
            "optional_b":        item_b.get('optional', False),
            "adjudicated_spans": None,
            "adjudication_note": "",
        })

    return {
        "meta": {
            "name_a":               name_1,
            "name_b":               name_2,
            "file_a":               file_a,
            "file_b":               file_b,
            "grammar_filter_stats": grammar_filter_stats,
            "stats":                stats,
        },
        "records": records,
    }
if __name__ == "__main__":
    base_dir = "anno"
    # file_1 = os.path.join(base_dir, f"annotations_wlj_smallmodel_163.json")
    # file_2 = os.path.join(base_dir, f"annotations_lyh_smallmodel_163.json")
    # output = "analysis/bn/annotators_analysis_smallmodel163.json"

    # file_1 = os.path.join(base_dir, f"annotations_single_gold_wlj.json")
    # file_2 = os.path.join(base_dir, f"annotations_single_gold_lyh.json")
    # output = "analysis/bn/annotators_analysis_gold113.json"

    # file_1 = os.path.join(base_dir, f"annotations_single_gold_random30_wlj.json")
    # file_2 = os.path.join(base_dir, f"annotations_single_gold_random30_lyh.json")
    # output = "analysis/bn/annotators_analysis_goldrandom30.json"

    # # o1mini认为正确
    # file_1 = os.path.join(base_dir, f"annotations_gold_o1right_random_wlj.json")
    # file_2 = os.path.join(base_dir, f"annotations_gold_o1right_random_lyh.json")
    # output = "analysis/bn/annotators_analysis_o1right_random30.json"


    # # o1mini和deepseek都认为是错误的：
    # file_1 = os.path.join(base_dir, f"annotations_smallmodel_botherror_random_wlj.json")
    # file_2 = os.path.join(base_dir, f"annotations_smallmodel_botherror_random_lyh.json")
    # output = "analysis/bn/annotators_analysis_botherror_random30.json"

    # # o1mini wrong和deepseek right：
    # file_1 = os.path.join(base_dir, f"annotations_single_gold_o1wrongdsright_93_wlj.json")
    # file_2 = os.path.join(base_dir, f"annotations_single_gold_o1wrongdsright_93_lyh.json")
    # output = "analysis/bn/annotators_analysis_single_gold_o1wrongdsright_93.json"


    # # mini wrong和deepseek right：
    # file_1 = os.path.join(base_dir, f"annotation_smallmodel_overlap_wlj.json")
    # file_2 = os.path.join(base_dir, f"annotation_smallmodel_overlap_lyh.json")
    # output = "analysis/bn/annotators_analysis_ssmallmodel_overlap_179.json"

    # # small model not recall的，不在final corrected data中,大模型判断错误的label
    # file_1 = os.path.join(base_dir, f"annotation_smnotrecall_21_wlj.json")
    # file_2 = os.path.join(base_dir, f"annotation_smnotrecall_21_lyh.json")
    # output = "analysis/bn/annotators_analysis_smnotrecall_21.json"

    # small model not recall的，不在final corrected data中的label，大模型判断正确的 抽查
    file_1 = os.path.join(base_dir, f"annotation_smnotrecallright_random30_wlj.json")
    file_2 = os.path.join(base_dir, f"annotation_smnotrecallright_random30_lyh.json")
    output = "analysis/bn/annotators_analysis_smnotrecallright_random30.json"

    file_wlj = read_json(file_1)
    file_lyh = read_json(file_2)
    

    # 步骤1：去重
    print("[去重]")
    data_a = remove_duplicates(file_wlj['annotations'])
    data_b = remove_duplicates(file_lyh['annotations'])
    raw_count_a, raw_count_b = len(data_a), len(data_b)
    print(f"  {file_1} 原始（去重后）：{raw_count_a} 条")
    print(f"  {file_2} 原始（去重后）：{raw_count_b} 条")
    
    # 步骤2: 抽取语法错误的数据：
    clean_wlj, error_wlj = filter_grammar_errors(file_wlj['annotations'])
    clean_lyh, error_lyh = filter_grammar_errors(file_lyh['annotations'])
    print(f"\n[语法错误过滤]")
    print(f"  {file_1} 认为句子正确个数：{len(clean_wlj)} 条，错误个数为 {len(error_wlj)} 条")
    print(f"  {file_2} 认为句子正确个数：{len(clean_lyh)} 条，错误个数为 {len(error_lyh)} 条")
    removed_idx = {item['idx'] for item in error_wlj} | {item['idx'] for item in error_lyh}

    grammar_filter_stats = {
        "removed_idx_count": len(removed_idx),
        "removed_idx_list":  sorted(removed_idx),
        "removed_by_wlj":      len(error_wlj),
        "removed_by_lyh":      len(error_lyh),
    }

    print(f"  涉及 idx 共 {len(removed_idx)} 个")
    if removed_idx:
        preview = sorted(removed_idx)[:20]
        suffix = f" ...（共 {len(removed_idx)} 个）" if len(removed_idx) > 20 else ""
        print(f"  被过滤的 idx：{preview}{suffix}")


    # 步骤3: 对齐数据（sen, prd, prd_idx, label）
    print("\n[对齐标注者数据]:")
    aligned_data = align_annotations(clean_wlj, clean_lyh)
    print(f"双方共同标注（语法正确）：{len(aligned_data)} 条")
     
    
    # 步骤4：计算匹配类型，抽取不一致
    print(f"正在处理: {file_1}  vs  {file_2}")
    results = extract(aligned_data, grammar_filter_stats, file_1, file_2, "wlj", "lyh" )
    write_json(results, output)
    s = results["meta"]["stats"]
    print(f"\n[结果汇总]")
    print(f"  共同标注（语法正确）：{s['total_aligned']} 条")
    print(f"  完全一致：{s.get('exact', 0) + s.get('both_empty', 0) +s.get('optional', 0)} 条")
    print(f"  不一致总计：{len(results['records'])} 条")
    print(f"    部分匹配：{s.get('partial', 0)}")
    print(f"    完全不匹配：{s.get('none', 0)+s.get('empty_b', 0)+s.get('empty_a', 0)}")
    print(f"        都有论元且完全不匹配：{s.get('none', 0)}")
    print(f"        仅 wlj 有标注：{s.get('empty_b', 0)}")
    print(f"        仅 lyh 有标注：{s.get('empty_a', 0)}")
    print(f"  完全一致的标注一致率：{s.get('exact', 0) + s.get('both_empty', 0)}/{s['total_aligned']} = {(s.get('exact', 0) + s.get('both_empty', 0))/s['total_aligned']:.2%}")
    print(f"  部分匹配的标注一致率：({s.get('exact', 0) + s.get('both_empty', 0)}+{s.get('partial', 0)})/{s['total_aligned']} = {(s.get('exact', 0) + s.get('both_empty', 0)+s.get('partial', 0))/s['total_aligned']:.2%}")
    print(f"  需要reveiw的个数为:{s['total_aligned'] - s.get('exact', 0) - s.get('both_empty', 0)}")
    print(s)
    print(f"\n输出文件：{output}")
    

