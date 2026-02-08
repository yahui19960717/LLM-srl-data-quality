# 给定句子、谓词和论元，构建LLM prompt来生成自然语言问题，

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

def stas_accuracy(data, path_save):
    total_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    total_gen_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    right_err_distribution = {}
    label_err_distribution = {}
    boundary_err_distribution = {}
    redundant_err_distribution = {}
    right_right_distribution = {}
    label_right_distribution = {}
    boundary_right_distribution = {}
    redundant_right_distribution = {}
    error_result = []
    for instance in data:
        gen_flag = instance.get('exist_relation', None)
        question = instance.get('question', '').strip()
        error_type = instance.get('error_type', None)
        select_span = instance.get('selected_span', None)
        temp_span = select_span[1].split("-")[-1]
        if error_type in total_num:
            total_num[error_type] += 1
            if gen_flag == 'yes' and question != '':
                total_gen_num[error_type] += 1
                if error_type == 'right':
                    right_right_distribution = add_dict(right_right_distribution, temp_span)
                elif error_type == 'label_error':
                    label_right_distribution = add_dict(label_right_distribution, temp_span)
                elif error_type == 'boundary_error':
                    boundary_right_distribution = add_dict(boundary_right_distribution, temp_span)
                elif error_type == 'redundant':
                    error_result.append(instance)
                    redundant_err_distribution = add_dict(redundant_err_distribution, temp_span)
            else:
                if error_type == 'right':
                    #error_result.append(instance)
                    right_err_distribution = add_dict(right_err_distribution, temp_span)
                elif error_type == 'boundary_error':
                    #error_result.append(instance)
                    boundary_err_distribution = add_dict(boundary_err_distribution, temp_span)
                elif error_type == 'label_error':
                    #error_result.append(instance)
                    label_err_distribution = add_dict(label_err_distribution, temp_span)
                elif error_type == 'redundant':
                    redundant_right_distribution = add_dict(redundant_right_distribution, temp_span)
    json.dump(error_result, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    print(f'总数分布：{total_num}')
    print(f'生成问题分布：{total_gen_num}')
    print(f'正确但未生成问题的角色类型分布：{right_err_distribution}')
    print(f'标签错误且未生成问题的角色类型分布：{label_err_distribution}')
    print(f'边界错误且未生成问题的角色类型分布：{boundary_err_distribution}')
    print(f'冗余且生成问题的角色类型分布：{redundant_err_distribution}')
    print(f'正确且生成问题的角色类型分布：{right_right_distribution}')
    print(f'标签错误但生成问题的角色类型分布：{label_right_distribution}')
    print(f'边界错误但生成问题的角色类型分布：{boundary_right_distribution}')
    print(f'冗余但未生成问题的角色类型分布：{redundant_right_distribution}')
    return 


# 统计问题生成的覆盖率和准确率
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    data = read_json('../llmout_question_generation/nw/nw-tc-test-ds-simp.json')
    path_out = 'question_generation_result/redundant_ds_error_qgen.json'
    stas_accuracy(data, path_out)

    """
    dataset = ['test'] #dev, 
    source = ['nw'] #["nw",  "bn", "bc" ]
    target = ['tc'] #['tc', 'bn', 'nw', 'bc']
    for k in dataset:
        for i in source:
            for j in target:
                data = read_json(f'../forllm_frames_newest/{i}/{i}-{j}-{k}.json')
                path_llmout = f'../llmout_question_generation/{i}/{i}-{j}-{k}-ds.json'
                path_llmout2 = f'../llmout_question_generation/{i}/{i}-{j}-{k}-ds-simp.json'
                LLM_prompt(data, path_llmout, path_llmout2)
                print("工作保存完成！")
    """
