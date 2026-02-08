# 构建prompt来应用LLM

import os
import sys
from sys import excepthook
from sympy import O
import torch
from tqdm import tqdm
import json
from openai import OpenAI
import numpy as np
import random
from collections import defaultdict, Counter
from config import Defination, read, write_json, read_json, labels_conll

def add_dict(dict_data, key_str):
    if key_str in dict_data:
        dict_data[key_str] += 1
    else:
        dict_data[key_str] = 1
    return dict_data

# 统计小模型中error_type的分布，包括在各角色类型上的分布,flag=0表示统计LLM判断的部分，flag=1表示frame部分数据
def stas_accuracy(data):
    right_distribution = {}
    label_error_distribution = {}
    boundary_error_distribution = {}
    redundant_distribution = {}
    
    for instance in tqdm(data):
        error_type = instance.get('error_type', None)
        select_span = instance.get('selected_span', None)
        candidate_labels = instance.get('candidate_roles', None)
        temp_span = select_span[1].split("-")[-1]
        #temp_span = select_span[1].split("-")[-1]
        if candidate_labels is not None and (len(candidate_labels) == 0 or temp_span not in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}): #不经过LLM判断了，直接设置为错误
            """
            if error_type == 'right':
                right_distribution = add_dict(right_distribution, temp_span)
            elif error_type == 'label_error':
                label_error_distribution = add_dict(label_error_distribution, temp_span)
            elif error_type == 'boundary_error':
                boundary_error_distribution = add_dict(boundary_error_distribution, temp_span)
            elif error_type == 'redundant':
                redundant_distribution = add_dict(redundant_distribution, temp_span)
            """
            continue
        
        if error_type == 'right':
            right_distribution = add_dict(right_distribution, temp_span)
        elif error_type == 'label_error':
            label_error_distribution = add_dict(label_error_distribution, temp_span)
        elif error_type == 'boundary_error':
            boundary_error_distribution = add_dict(boundary_error_distribution, temp_span)
        elif error_type == 'redundant':
            redundant_distribution = add_dict(redundant_distribution, temp_span)
        
        

    print(f'正确分布：{right_distribution}')
    print(f'标签错误分布：{label_error_distribution}')
    print(f'边界错误分布：{boundary_error_distribution}')
    print(f'冗余错误分布：{redundant_distribution}')

# 统计大模型判断情况，参考error_type的分布和大模型判断情况，统计大模型判断正确与否在各类角色上的分布
def stas_llm_accuracy(data, path_save):
    right_distribution_v1 = {}
    label_error_distribution_v1 = {}
    boundary_error_distribution_v1 = {}
    redundant_distribution_v1 = {}
    right_distribution_v2 = {}
    label_error_distribution_v2 = {}
    boundary_error_distribution_v2 = {}
    redundant_distribution_v2 = {}
    error_list = []
    
    for instance in tqdm(data):
        error_type = instance.get('error_type', None)
        final_judgment = instance.get('final_judgement', None)
        if final_judgment is None: #未经过LLM判断
            continue
        select_span = instance.get('selected_span', None)
        temp_span = select_span[1].split("-")[-1]
        #temp_span = select_span[1].split("-")[-1]
        if error_type == 'right':
            if final_judgment == 'correct':
                right_distribution_v1 = add_dict(right_distribution_v1, temp_span)
            else:
                right_distribution_v2 = add_dict(right_distribution_v2, temp_span)
                #error_list.append(instance)
        elif error_type == 'label_error':
            if final_judgment == 'incorrect':
                label_error_distribution_v1 = add_dict(label_error_distribution_v1, temp_span)
            else:
                label_error_distribution_v2 = add_dict(label_error_distribution_v2, temp_span)
                error_list.append(instance)
        elif error_type == 'boundary_error':
            if final_judgment == 'incorrect':
                boundary_error_distribution_v1 = add_dict(boundary_error_distribution_v1, temp_span)
            else:
                boundary_error_distribution_v2 = add_dict(boundary_error_distribution_v2, temp_span)
                #error_list.append(instance)
        elif error_type == 'redundant':
            if final_judgment == 'incorrect':
                redundant_distribution_v1 = add_dict(redundant_distribution_v1, temp_span)
            else:
                redundant_distribution_v2 = add_dict(redundant_distribution_v2, temp_span)
                #error_list.append(instance)
    print(f'正确分布-LLM判断正确：{right_distribution_v1}')
    print(f'正确分布-LLM判断错误：{right_distribution_v2}')
    print(f'标签错误分布-LLM判断正确：{label_error_distribution_v1}')
    print(f'标签错误分布-LLM判断错误：{label_error_distribution_v2}')
    print(f'边界错误分布-LLM判断正确：{boundary_error_distribution_v1}')
    print(f'边界错误分布-LLM判断错误：{boundary_error_distribution_v2}')
    print(f'冗余错误分布-LLM判断正确：{redundant_distribution_v1}')
    print(f'冗余错误分布-LLM判断错误：{redundant_distribution_v2}')
    json.dump(error_list, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    #data_source = read_json('../forllm_frames_newest/nw/nw-tc-test.json')
    #stas_accuracy(data_source)
    data_o1 = read_json('../llmout_lyh/nw/nw-bn-test-llmsimp.json')
    path_out = 'nw_bn_label_o1_error.json'
    stas_llm_accuracy(data_o1, path_out)
    

    
