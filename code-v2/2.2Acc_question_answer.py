# 统计问题回答准确率

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
from collections import defaultdict, Counter
from config import Defination, read, write_json, read_json, labels_conll

def add_dict(dict_data, key_str):
    if key_str in dict_data:
        dict_data[key_str] += 1
    else:
        dict_data[key_str] = 1
    return dict_data

def safe_json_loads(field):
    if isinstance(field, str):
        return json.loads(field)
    elif isinstance(field, dict):
        return field  # 已经是字典，直接返回
    else:
        raise ValueError(f"Unexpected type for JSON field: {type(field)}")


def stas_core_accuracy(data):
    #不同错误类型，未产出答案和错误答案的分布
    right_err_distribution = {}
    label_err_distribution = {}
    boundary_err_distribution = {}
    redundant_err_distribution = {} 
    # 不同错误类型下答案正确的分布
    right_right_distribution = {}
    label_right_distribution = {}
    boundary_right_distribution = {}
    redundant_right_distribution = {} 

    total_num = 0
    core_total_num = 0
    nocore_total_num = 0

    ori_correct_num = 0
    ori_core_correct_num = 0
    ori_nocore_correct_num = 0
    llm_correct_num = 0
    llm_core_correct_num = 0
    llm_nocore_correct_num = 0
    

    for instance in data:
        qus_response = safe_json_loads(instance["response"])
        ans_response = safe_json_loads(instance["ans_response"])
        gen_flag = qus_response.get('exist_relation', None)
        question = qus_response.get('question', '').strip()
        has_answer = ans_response.get('has_answer', None)
        answer = ans_response.get('answer', '').strip()
        error_type = instance.get('error_type', None)
        select_span = instance.get('selected_span', None)
        temp_span = select_span[1].split("-")[-1]

        if gen_flag != 'yes' or question == '':
            continue
        total_num += 1
        arg_type = 'nocore'
        if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
            arg_type = 'core'
            core_total_num += 1
            if error_type == 'right':
                ori_correct_num += 1
                ori_core_correct_num += 1
        else:
            nocore_total_num += 1
            if error_type == 'right':
                ori_correct_num += 1
                ori_nocore_correct_num += 1
        if error_type == 'right' and answer == select_span[0]:
            llm_correct_num += 1
            if arg_type == 'core':
                llm_core_correct_num += 1
            else:
                llm_nocore_correct_num += 1
        elif error_type != 'right' and (has_answer != 'yes' or answer == '' or answer != select_span[0]):
            llm_correct_num += 1
            if arg_type == 'core':
                llm_core_correct_num += 1
            else:
                llm_nocore_correct_num += 1

        if has_answer != 'yes' or answer == '':
            if error_type == 'right':
                right_err_distribution = add_dict(right_err_distribution, arg_type)
            elif error_type == 'label_error':
                label_right_distribution = add_dict(label_right_distribution, arg_type)
            elif error_type == 'boundary_error':
                boundary_right_distribution = add_dict(boundary_right_distribution, arg_type)
            elif error_type == 'redundant':
                redundant_right_distribution = add_dict(redundant_right_distribution, arg_type)
        elif answer == select_span[0]: #答案正确
            if error_type == 'right':
                right_right_distribution = add_dict(right_right_distribution, arg_type)
            elif error_type == 'label_error':
                label_err_distribution = add_dict(label_err_distribution, arg_type)
            elif error_type == 'boundary_error':
                boundary_err_distribution = add_dict(boundary_err_distribution, arg_type)
            elif error_type == 'redundant':
                redundant_err_distribution = add_dict(redundant_err_distribution, arg_type)
        else: #答案错误
            if error_type == 'right':
                right_err_distribution = add_dict(right_err_distribution, arg_type)
            elif error_type == 'label_error':
                label_right_distribution = add_dict(label_right_distribution, arg_type)
            elif error_type == 'boundary_error':
                boundary_right_distribution = add_dict(boundary_right_distribution, arg_type)
            elif error_type == 'redundant':
                redundant_right_distribution = add_dict(redundant_right_distribution, arg_type)
    print(f'正确论元，未产出答案或答案错误的分布：{right_err_distribution}')
    print(f'关系错误论元，未产出答案或答案错误的分布：{label_err_distribution}')
    print(f'边界错误论元，未产出答案或答案错误的分布：{boundary_err_distribution}')
    print(f'冗余识别论元，产出与候选匹配答案的分布：{redundant_err_distribution}')
    print(f'正确论元，答案正确的分布：{right_right_distribution}')
    print(f'关系错误论元，答案正确的分布：{label_right_distribution}')
    print(f'边界错误论元，答案正确的分布：{boundary_right_distribution}')
    print(f'冗余识别论元，未产出答案或答案错误的分布：{redundant_right_distribution}')  

    print(f'总数：{total_num}, 核心论元总数：{core_total_num}, 非核心论元总数：{nocore_total_num}')
    print(f'原始正确数：{ori_correct_num}, 原始核心论元正确数：{ori_core_correct_num}, 原始非核心论元正确数：{ori_nocore_correct_num}')
    print(f'LLM正确数：{llm_correct_num}, LLM核心论元正确数：{llm_core_correct_num}, LLM非核心论元正确数：{llm_nocore_correct_num}')

    ori_acc = ori_correct_num / total_num if total_num > 0 else 0
    ori_core_acc = ori_core_correct_num / core_total_num if core_total_num > 0 else 0
    ori_nocore_acc = ori_nocore_correct_num / nocore_total_num if nocore_total_num > 0 else 0
    llm_acc = llm_correct_num / total_num if total_num > 0 else 0
    llm_core_acc = llm_core_correct_num / core_total_num if core_total_num > 0 else 0
    llm_nocore_acc = llm_nocore_correct_num / nocore_total_num if nocore_total_num > 0 else 0
    print(f'原始整体准确率：{ori_acc}, 原始核心论元准确率：{ori_core_acc}, 原始非核心论元准确率：{ori_nocore_acc}')
    print(f'LLM整体准确率：{llm_acc}, LLM核心论元准确率：{llm_core_acc}, LLM非核心论元准确率：{llm_nocore_acc}')
    return 



def stas_accuracy(data, path_save1, path_save2):
    total_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    total_gen_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    correct_ans_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    incorrect_ans_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    error_ans_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}

    #不同错误类型，未产出答案的分布
    right_err_distribution = {}
    label_err_distribution = {}
    boundary_err_distribution = {}
    redundant_err_distribution = {} 
    # 不同错误类型下答案正确的分布
    right_right_distribution = {}
    label_right_distribution = {}
    boundary_right_distribution = {}
    redundant_right_distribution = {} 
    # 不同错误类型下，有答案但答案错误的分布
    right_incorrect_distribution = {}
    label_incorrect_distribution = {}
    boundary_incorrect_distribution = {}
    redundant_incorrect_distribution = {}

    error_result = [] #未产出答案数据
    incorrect_result = [] #错误答案数据

    for instance in data:
        #qus_response = json.loads(instance["response"])
        #ans_response = json.loads(instance["ans_response"])
        qus_response = safe_json_loads(instance["response"])
        ans_response = safe_json_loads(instance["ans_response"])
        gen_flag = qus_response.get('exist_relation', None)
        question = qus_response.get('question', '').strip()
        has_answer = ans_response.get('has_answer', None)
        answer = ans_response.get('answer', '').strip()
        error_type = instance.get('error_type', None)
        select_span = instance.get('selected_span', None)
        temp_span = select_span[1].split("-")[-1]

        if error_type not in total_num:
            continue
        total_num[error_type] += 1
        if gen_flag == 'yes' and question != '':
            total_gen_num[error_type] += 1
            if has_answer == 'yes' and answer == select_span[0]: #有答案 且 答案正确
                correct_ans_num[error_type] += 1
                if error_type == 'right':
                    right_right_distribution = add_dict(right_right_distribution, temp_span)
                elif error_type == 'label_error':
                    label_right_distribution = add_dict(label_right_distribution, temp_span)
                elif error_type == 'boundary_error':
                    boundary_right_distribution = add_dict(boundary_right_distribution, temp_span)
                elif error_type == 'redundant':
                    #error_result.append(instance)
                    redundant_err_distribution = add_dict(redundant_err_distribution, temp_span)
            elif has_answer == 'yes' and answer != '' and answer != select_span[0]: #有答案 但答案错误
                incorrect_ans_num[error_type] += 1
                if error_type == 'right':
                    incorrect_result.append(instance)
                    right_incorrect_distribution = add_dict(right_incorrect_distribution, temp_span)
                elif error_type == 'label_error':
                    #incorrect_result.append(instance)
                    label_incorrect_distribution = add_dict(label_incorrect_distribution, temp_span)
                elif error_type == 'boundary_error':
                    #incorrect_result.append(instance)
                    boundary_incorrect_distribution = add_dict(boundary_incorrect_distribution, temp_span)
                elif error_type == 'redundant':
                    #incorrect_result.append(instance)
                    redundant_incorrect_distribution = add_dict(redundant_incorrect_distribution, temp_span)
            else: #无答案
                error_ans_num[error_type] += 1
                if error_type == 'right':
                    error_result.append(instance)
                    right_err_distribution = add_dict(right_err_distribution, temp_span)
                elif error_type == 'boundary_error':
                    #error_result.append(instance)
                    boundary_err_distribution = add_dict(boundary_err_distribution, temp_span)
                elif error_type == 'label_error':
                    #error_result.append(instance)
                    label_err_distribution = add_dict(label_err_distribution, temp_span)
                elif error_type == 'redundant':
                    redundant_right_distribution = add_dict(redundant_right_distribution, temp_span)

    print(f'整体数据分布：{total_num}')
    print(f'生成问题数据分布：{total_gen_num}')
    print(f'正确答案数据分布：{correct_ans_num}')
    print(f'错误答案数据分布：{incorrect_ans_num}')
    print(f'无答案数据分布：{error_ans_num}')
    print(f'正确论元，未产出答案的角色类型分布：{right_err_distribution}')
    print(f'关系错误论元，未产出答案的角色类型分布：{label_err_distribution}')
    print(f'边界错误论元，未产出答案的角色类型分布：{boundary_err_distribution}')
    print(f'冗余识别论元，产出与候选匹配答案的角色类型分布：{redundant_err_distribution}')
    print(f'正确论元，答案正确的角色类型分布：{right_right_distribution}')
    print(f'关系错误论元，答案正确的角色类型分布：{label_right_distribution}')
    print(f'边界错误论元，答案正确的角色类型分布：{boundary_right_distribution}')
    print(f'冗余识别论元，未产出答案的角色类型分布：{redundant_right_distribution}')
    print(f'正确论元，产出答案与候选不匹配的角色类型分布：{right_incorrect_distribution}')
    print(f'关系错误论元，产出答案与候选不匹配的角色类型分布：{label_incorrect_distribution}')
    print(f'边界错误论元，产出答案与候选不匹配的角色类型分布：{boundary_incorrect_distribution}')
    print(f'冗余识别论元，产出答案与候选不匹配的角色类型分布：{redundant_incorrect_distribution}')    
    
    json.dump(error_result, open(path_save1, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    json.dump(incorrect_result, open(path_save2, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    return


def stas_accuracy_for_pattern_question(data, path_save1, path_save2):
    total_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    correct_ans_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    incorrect_ans_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}
    error_ans_num = {'right':0, 'label_error':0, 'boundary_error':0, 'redundant':0}

    #不同错误类型，未产出答案的分布
    right_err_distribution = {}
    label_err_distribution = {}
    boundary_err_distribution = {}
    redundant_err_distribution = {} 
    # 不同错误类型下答案正确的分布
    right_right_distribution = {}
    label_right_distribution = {}
    boundary_right_distribution = {}
    redundant_right_distribution = {} 
    # 不同错误类型下，有答案但答案错误的分布
    right_incorrect_distribution = {}
    label_incorrect_distribution = {}
    boundary_incorrect_distribution = {}
    redundant_incorrect_distribution = {}

    error_result = [] #未产出答案数据
    incorrect_result = [] #错误答案数据

    for instance in data:
        error_type = instance.get('error_type', None)
        select_span = instance.get('selected_span', None)
        temp_span = select_span[1].split("-")[-1]
        has_answer = instance.get('has_answer', None)
        answer = instance.get('answer', None)

        if error_type not in total_num:
            continue
        total_num[error_type] += 1
        if has_answer == 'yes' and answer == select_span[0]: #有答案 且 答案正确
            correct_ans_num[error_type] += 1
            if error_type == 'right':
                right_right_distribution = add_dict(right_right_distribution, temp_span)
            elif error_type == 'label_error':
                label_right_distribution = add_dict(label_right_distribution, temp_span)
            elif error_type == 'boundary_error':
                boundary_right_distribution = add_dict(boundary_right_distribution, temp_span)
            elif error_type == 'redundant':
                error_result.append(instance)
                redundant_err_distribution = add_dict(redundant_err_distribution, temp_span)
        elif has_answer == 'yes' and answer != '' and answer != select_span[0]: #有答案 但答案错误
            incorrect_ans_num[error_type] += 1
            if error_type == 'right':
                incorrect_result.append(instance)
                right_incorrect_distribution = add_dict(right_incorrect_distribution, temp_span)
            elif error_type == 'label_error':
                incorrect_result.append(instance)
                label_incorrect_distribution = add_dict(label_incorrect_distribution, temp_span)
            elif error_type == 'boundary_error':
                incorrect_result.append(instance)
                boundary_incorrect_distribution = add_dict(boundary_incorrect_distribution, temp_span)
            elif error_type == 'redundant':
                incorrect_result.append(instance)
                redundant_incorrect_distribution = add_dict(redundant_incorrect_distribution, temp_span)
        else: #无答案
            error_ans_num[error_type] += 1
            if error_type == 'right':
                error_result.append(instance)
                right_err_distribution = add_dict(right_err_distribution, temp_span)
            elif error_type == 'boundary_error':
                error_result.append(instance)
                boundary_err_distribution = add_dict(boundary_err_distribution, temp_span)
            elif error_type == 'label_error':
                error_result.append(instance)
                label_err_distribution = add_dict(label_err_distribution, temp_span)
            elif error_type == 'redundant':
                redundant_right_distribution = add_dict(redundant_right_distribution, temp_span)

    print(f'整体数据分布：{total_num}')
    print(f'正确答案数据分布：{correct_ans_num}')
    print(f'错误答案数据分布：{incorrect_ans_num}')
    print(f'无答案数据分布：{error_ans_num}')
    print(f'正确论元，未产出答案的角色类型分布：{right_err_distribution}')
    print(f'关系错误论元，未产出答案的角色类型分布：{label_err_distribution}')
    print(f'边界错误论元，未产出答案的角色类型分布：{boundary_err_distribution}')
    print(f'冗余识别论元，产出与候选匹配答案的角色类型分布：{redundant_err_distribution}')
    print(f'正确论元，答案正确的角色类型分布：{right_right_distribution}')
    print(f'关系错误论元，答案正确的角色类型分布：{label_right_distribution}')
    print(f'边界错误论元，答案正确的角色类型分布：{boundary_right_distribution}')
    print(f'冗余识别论元，未产出答案的角色类型分布：{redundant_right_distribution}')
    print(f'正确论元，产出答案与候选不匹配的角色类型分布：{right_incorrect_distribution}')
    print(f'关系错误论元，产出答案与候选不匹配的角色类型分布：{label_incorrect_distribution}')
    print(f'边界错误论元，产出答案与候选不匹配的角色类型分布：{boundary_incorrect_distribution}')
    print(f'冗余识别论元，产出答案与候选不匹配的角色类型分布：{redundant_incorrect_distribution}')    
    
    json.dump(error_result, open(path_save1, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    json.dump(incorrect_result, open(path_save2, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    return




# 训练遍历TEST下每一个文件，针对每一个边缘概率对应角色进行问题生成
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    #data = read_json(f'../llmout_question_answer/nw/nw-tc-test-ds-o1.json')
    #path_llmout = f'question_answer_result/right_o1_no_ans.json'
    #path_llmout2 = f'question_answer_result/right_o1_incorrect_ans.json'
    #stas_accuracy(data, path_llmout, path_llmout2)
    #stas_core_accuracy(data)
    data = read_json(f'question_generation_result/nw-bn-test-pattern-o1-simp.json')
    path_llmout = f'question_generation_result/nw-bn-test-pattern-o1-noans.0212.json'
    path_llmout2 = f'question_generation_result/nw-bn-test-pattern-o1-incorrectans.0212.json'
    stas_accuracy_for_pattern_question(data, path_llmout, path_llmout2)
