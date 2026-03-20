# 构建prompt来应用LLM

import os
import sys
#print(sys.executable)
from sys import excepthook
from sympy import O
import torch
#print(torch.__file__)
from tqdm import tqdm
import json
from openai import OpenAI
import numpy as np
import random
from collections import defaultdict, Counter
from config import Defination, read, write_json, read_json, labels_conll
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']

random.seed(42)
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",#tcy
    api_key="sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz" # tcy
)


def check_error(instance):
    sen_id = instance['index_sen'] #句子id
    sentence = instance['sentences'] #句子字符串
    sen_span = instance['selected_span'] #候选论元信息 (含论元字符串, 角色标签, 论元在句子中的位置，起始位置、终止位置+1)
    error_type = instance['error_type'] #错误类型，根据golden结果和预测结果生成的，含关系错误、边界错误、正确、多余
    org_span = instance["org_span"]  # predict论元在句子xx，谓词id+1，起始位置、终止位置+1，xx
    conflict_span = instance["conflict_span"]  # 和预测冲突的论元 golden论元在句子xx，谓词id+1，起始位置、终止位置+1，xx
    if error_type == "boundary_error":
        if sen_span[2][0] == conflict_span[2] and sen_span[2][1] == conflict_span[3]:  #预测边界跟golden边界一致，应该是数据处理错误
            return True
    return False

def check_data(data, path_save):
    path = "/".join(path_save.split("/")[:-1])
    # import pdb;pdb.set_trace()
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"路径已创建: {path}")
    else:
        print(f"路径已存在: {path}")
    error_instance_list = []
    error_num = 0
    boundary_error_num = 0
    for instance in tqdm(data): 
        if instance['error_type'] == "boundary_error":
            boundary_error_num += 1
        flag = check_error(instance)
        if flag == True:
            error_num += 1
            error_instance_list.append(instance)
    print(error_num, boundary_error_num)
    json.dump(error_instance_list, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 



if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 
    source = ["nw",  "bn", "bc" ]# 'bn'
    target = ['tc', 'bn', 'nw', 'bc'] # 'tc',
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                data = read_json(f'../forllm_frames_newest/{i}/{i}-{j}-{k}.json')
                path_llmout = f'../llmout_lyh/{i}/{i}-{j}-{k}-llm.json'
                path_error = f'../llmout_lyh/{i}/{i}-{j}-{k}-error.json'
                check_data(data, path_error)
                print("数据检查完成！")
            
    
