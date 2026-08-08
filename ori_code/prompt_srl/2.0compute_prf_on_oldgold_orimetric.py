#!/usr/bin/env python3

import os
import sys
import time
from sys import excepthook
from sympy import O
import torch
from tqdm import tqdm
import json
from openai import OpenAI
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

# step1:
# 写一个函数parse_llm_result，输入参数为读取文件名称，返回一个字典
# 文件中每一条数据格式为："sent\tprd_word\tprd_idx":{"prd_sense":"", "roles":{}, "prompt":"", "Prompt_Result":""}", 其中Prompt_Result字段的格式为{"comment":{"role1":argument1,"role2":argument2,...}}
# 解析输入中的Prompt_Result结果，保存每一个role对应的Argument；最后字典的key为sent\tprd_word\tprd_idx，value为一个字典，以role为key、Argument为value
def parse_llm_result(data): 
    result_dict = {}
    print_num = 0
    for key, value in data.items():
        sent, prd_word, prd_idx = key.split("\t")
        prompt_result = value.get("Prompt_Result", {})
        
        # 处理 Prompt_Result 是字符串的情况（可能是 JSON 字符串）
        if isinstance(prompt_result, str):
            try:
                # 尝试解析 JSON 字符串
                prompt_result = json.loads(prompt_result)
            except json.JSONDecodeError:
                # 如果不是有效的 JSON，尝试简单的格式处理，或者保持为空字典
                print(f"Warning: Could not parse Prompt_Result as JSON for key: {key}")
                prompt_result = {}
        
        comment_data = prompt_result.get("comment", {}) if isinstance(prompt_result, dict) else {}
        #print(comment_data)
        role_argument_dict = {}
        if isinstance(comment_data, dict):
            for role, argument in comment_data.items():
                role_argument_dict[role] = argument
        result_dict[key] = role_argument_dict
        #if print_num < 5:
        #    print(key,result_dict[key])
        #print_num += 1
    return result_dict


# step2：写一个函数parse_gold_annotation, 输入为golden数据，输出为一个字典；
# golden的每一条数据是一个json文件，有sen、prd_word、prd_idx、prd_lemma、label、span和span_idx等字段，以sen、prd_word、prd_idx为key，整合该key下label和span；
# 注意，一条数据仅有一个label和span，同一个sen、prd_word、prd_idx的label和span会存在多条数据中，需要都整合到一个字典中
# 返回的字典以sen、prd_word、prd_idx为key，value为一个字典，以label为key、span为value
def parse_gold_annotation(data):
    result_dict = {}
    print_num = 0
    for item in data:
        sent = item["sen"]
        prd_word = item["prd_word"]
        prd_idx = item["prd_idx"]
        key = f"{sent}\t{prd_word}\t{prd_idx}"
        label = item["label"]
        span = item["span"]
        if key not in result_dict:
            result_dict[key] = {label: span}
        else:
            result_dict[key][label] = span
        #if print_num < 5:
        #    print(key,result_dict[key])
        #print_num += 1
    return result_dict

def normalize_span(span):
    """将 span 统一转换为小写字符串，处理列表、数字或 None 等情况"""
    if span is None:
        return ""
    if isinstance(span, list):
        # 如果是列表，将其元素（如单词）连接成字符串
        return " ".join([str(item) for item in span]).lower().strip()
    return str(span).lower().strip()

def compute_prf(gold_dict, llm_dict):
    role_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) #tp真正例，fp假正例，fn假负例
    
    # 获取所有的 keys (sent\tprd_word\tprd_idx)
    all_keys = set(gold_dict.keys()) | set(llm_dict.keys())
    
    for key in all_keys:
        gold_roles = gold_dict.get(key, {})
        llm_roles = llm_dict.get(key, {})
        
        # 所有的角色类型
        all_role_types = set(gold_roles.keys()) | set(llm_roles.keys())
        
        for role in all_role_types:
            gold_span = gold_roles.get(role)
            llm_span = llm_roles.get(role)
            
            # 统一处理 span 的格式
            norm_gold = normalize_span(gold_span)
            norm_llm = normalize_span(llm_span)
            
            if norm_gold and norm_llm:
                if norm_gold == norm_llm:  
                    role_metrics[role]["tp"] += 1
                else:
                    role_metrics[role]["fp"] += 1
                    role_metrics[role]["fn"] += 1
            elif norm_gold:
                # gold 有，llm 没有 -> fn
                role_metrics[role]["fn"] += 1
            elif norm_llm:
                # llm 有，gold 没有 -> fp
                role_metrics[role]["fp"] += 1
                
    # 计算每个角色的 P, R, F1
    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for role, counts in role_metrics.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[role] = {"precision": precision, "recall": recall, "f1": f1}
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    # 计算整体指标
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    results["OVERALL"] = {"precision": overall_p, "recall": overall_r, "f1": overall_f1}
    
    return results

# step3：以第二步获得的数据为标准，计算第一步文件在各个角色上的precision、recall和F1，以及最后给出整体角色上的precision、recall和F1
if __name__ == "__main__":
    domain = "bn" #"bn"
    model = "gpt4.1"
    
    # sstep1:先读取llm_result/test_bn_4llm_core_gold_role.json，解析Prompt_Result结果，保存每一个role对应的Argument
    llm_path = f"llm_result/{model}_result/test_{domain}_4llm_core_gold_role.json"
    llm_data = read_json(llm_path)
    print("process llm result\n")
    llm_role_argument_dict = parse_llm_result(llm_data)
        
    # step2：读取原始gold数据，解析gold结果，保存每一个role对应的Argument
    gold_path = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold.conll"
    gold_data = read_json(gold_path)
    print("process gold result\n")
    gold_role_argument_dict = parse_gold_annotation(gold_data)
            
    # step3: 计算 PRF
    results = compute_prf(gold_role_argument_dict, llm_role_argument_dict)
    print("\nEvaluation Results:")
    for role, metrics in results.items():
        # 仅打印role是ARG0、ARG1、ARG2、ARG3、ARG4、ARG5和OVERALL的结果
        if role in ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "OVERALL"]:
            print(f"Role: {role:10} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    
    
