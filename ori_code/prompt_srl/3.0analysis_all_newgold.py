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
                if argument == "" or argument == None or argument == "None":
                    continue
                role_argument_dict[role] = argument
        result_dict[key] = role_argument_dict
        #if print_num < 5:
        #    print(key,result_dict[key])
        #print_num += 1
    return result_dict


# step2：写一个函数parse_gold_annotation, 输入为golden数据，输出为一个字典；
# golden的每一条数据是一个json文件，'idx','sentence', 'prd_word', 'prd_idx', 'label', 'span_mean', 'type', 'selected_spans'，其中selected_spans含有字段'start', 'end', 'text';
# 以sentense、prd_word、prd_idx为key，整合该key下label和selected_spans，这里一个label可能对应多个span；
# 注意，一条数据仅有一个label和selected_spans，同一个sen、prd_word、prd_idx的label和span会存在多条数据中，需要都整合到一个字典中
# 返回的字典以sen、prd_word、prd_idx为key，value为一个字典，以label为key、span为value
def parse_gold_annotation(data):
    result_dict = {}
    opt_result_dict = {}
    label_mean_dict = {}
    print_num = 0
    for iter_key, item in data.items():
        sent = item["sentence"]
        prd_word = item["prd_word"]
        prd_idx = item["prd_idx"]
        key = f"{sent}\t{prd_word}\t{prd_idx}"
        label = item["label"]

        span_mean = item["span_mean"]
        key_mean = f"{key}\t{label}"
        if key_mean not in label_mean_dict:
            label_mean_dict[key_mean] = span_mean

        opt_flag = item.get("optional", False)
        selected_spans = item["selected_spans"]
        span_list = []
        for span in selected_spans:
            span_list.append(span["text"])
        if span_list == []:
            continue
        if key not in result_dict:
            result_dict[key] = {label: span_list}
        else:
            result_dict[key][label] = span_list
        #单独记录可标可不标的label和对应的span
        opt_key = f"{key}\t{label}"
        if opt_flag == True or opt_flag == "true":
            opt_result_dict[opt_key] = span_list
        
        #if print_num < 5:
        #    print(key,result_dict[key])
        #print_num += 1
    return result_dict, opt_result_dict, label_mean_dict

def normalize_span(span):
    """将 span 统一转换为小写字符串，处理列表、数字或 None 等情况"""
    if span is None:
        return ""
    if isinstance(span, list):
        # 如果是列表，将其元素（如单词）连接成字符串
        return " ".join([str(item) for item in span]).lower().strip()
    return str(span).lower().strip()

def match_span(span_list, span):
    jieci_list = ["in ", "on ", "at ", "by ", "with ", "for ", "to ", "from ", "down ", "up ", "around ", "near ", "as ", "under ", "over ", "between ", "above ", "below "]
    for span_item in span_list:
        for jieci in jieci_list:
            if span_item.startswith(jieci):
                span_item = span_item[jieci.__len__():]
        if span_item == span:
            return True
        span_item_new = span_item.replace(" ", "")
        span_new = span.replace(" ", "")
        if span_item_new == span_new:
            return True
    return False

def analysis_error(gold_dict, llm_dict, opt_gold_dict, label_mean_dict):
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
            key_mean = f"{key}\t{role}"
            span_mean = label_mean_dict.get(key_mean, None)
            
            # 统一处理 span 的格式，对于gold_span，其是列表，需要对列表的每一个元素进行处理，最后还是返回列表
            norm_gold = [normalize_span(span) for span in gold_span] if gold_span else []
            # 对于llm_span，其是字符串，需要直接处理
            norm_llm = normalize_span(llm_span)
            gold_str = " ".join(norm_gold)

            opt_key = f"{key}\t{role}"
            
            if norm_gold and norm_llm:
                if norm_llm not in norm_gold and not match_span(norm_gold, norm_llm):  
                    # 输出key，role，norm_llm，norm_gold，按着tab键隔开
                    error_type = "boundary_error"
                    print(f"{key}\t{role}\t{span_mean}\t{norm_llm}\t{gold_str}\t{error_type}")
            elif norm_gold:
                # gold 有，llm 没有 -> fn
                if opt_key not in opt_gold_dict:
                    error_type = "miss_error"
                    print(f"{key}\t{role}\t{span_mean}\t{norm_llm}\t{gold_str}\t{error_type}")
            elif norm_llm:
                # llm 有，gold 没有 -> fp
                if opt_key not in opt_gold_dict:
                    error_type = "extra_error"
                    print(f"{key}\t{role}\t{span_mean}\t{norm_llm}\t{gold_str}\t{error_type}")
    return

# step3：以第二步获得的数据为标准，计算第一步文件在各个角色上的precision、recall和F1，以及最后给出整体角色上的precision、recall和F1
if __name__ == "__main__":
    domain = "tc" #"tc"
    
    # sstep1:先读取llm_result/test_bn_4llm_core_gold_role.json，解析Prompt_Result结果，保存每一个role对应的Argument
    llm_path = f"llm_result/test_{domain}_4llm_core_gold_role.json"
    llm_data = read_json(llm_path)
    #print("process llm result\n")
    llm_role_argument_dict = parse_llm_result(llm_data)
        
    # step2：读取/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/new_test_set/test_bn_500_core_final.json
    gold_path = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/new_test_set/test_{domain}_500_core_final.json"
    gold_data = read_json(gold_path)
    #print("process gold result\n")
    gold_role_argument_dict, opt_gold_role_argument_dict, label_mean_dict = parse_gold_annotation(gold_data)
            
    # step3: 分析错误类型
    analysis_error(gold_role_argument_dict, llm_role_argument_dict, opt_gold_role_argument_dict, label_mean_dict) 
    
