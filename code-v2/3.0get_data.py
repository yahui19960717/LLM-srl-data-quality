# 获取指定数据

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

def safe_json_loads(field):
    if isinstance(field, str):
        return json.loads(field)
    elif isinstance(field, dict):
        return field  # 已经是字典，直接返回
    else:
        raise ValueError(f"Unexpected type for JSON field: {type(field)}")

def transfer_json_to_tab(data):
    
    # 按error type选择指定类型的数据，保存在data_list中
    for instance in tqdm(data):
        index_sen = instance.get('index_sen', None)
        sentence = instance.get('sentences', None)
        predicate = instance.get('predicate', None)
        select_span = instance.get('selected_span', None)
        qus_response = safe_json_loads(instance["response"])
        ans_response = safe_json_loads(instance["ans_response"])
        gen_flag = qus_response.get('exist_relation', None)
        question = qus_response.get('question', '').strip()
        has_answer = ans_response.get('has_answer', None)
        answer = ans_response.get('answer', '').strip()
        error_type = instance.get('error_type', None)
        gold_label = instance.get('gold_label', None)

        
        print(f'{index_sen} \t {sentence} \t {predicate} \t {select_span[0]} \t {select_span[1]} \t {error_type} \t {gen_flag} \t {question} \t {has_answer} \t {answer} \t {gold_label}')
    
def transfer_json_to_tab_v2(data):
    
    # 按error type选择指定类型的数据，保存在data_list中
    for instance in tqdm(data):
        index_sen = instance.get('index_sen', None)
        sentence = instance.get('sentences', None)
        predicate = instance.get('predicate', None)
        select_span = instance.get('selected_span', None)
        error_type = instance.get('error_type', None)
        gold_label = instance.get('gold_label', None)
        final_judgement = instance.get('final_judgement', None)
        confict_span = instance.get('conflict_span', None)

        
        print(f'{index_sen} \t {sentence} \t {predicate} \t {select_span[0]} \t {select_span[1]} \t {error_type} \t {final_judgement} \t {gold_label} \t {confict_span}')
    


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    
    #data = read_json(f'question_answer_result/boundary_o1_incorrect_ans.json')
    #data = read_json(f'question_answer_result/boundary_o1_no_ans.json')

    #data = read_json(f'question_answer_result/label_o1_incorrect_ans.json')
    #data = read_json(f'question_answer_result/label_o1_no_ans.json')
    #data = read_json(f'question_answer_result/redundant_o1_incorrect_ans.json')
    #data = read_json(f'question_answer_result/redundant_o1_no_ans.json')
    #data = read_json(f'question_answer_result/right_o1_incorrect_ans.json')
    #data = read_json(f'question_answer_result/right_o1_no_ans.json')
    #transfer_json_to_tab(data)
    
    #data = read_json(f'o1_error_result/nw_bn_boundary_o1_error.json')
    #data = read_json(f'o1_error_result/nw_bn_label_o1_error.json')
    data = read_json(f'o1_error_result/nw_bn_redundant_o1_error.json')
    #data = read_json(f'o1_error_result/nw_bn_right_o1_error.json')
    transfer_json_to_tab_v2(data)
