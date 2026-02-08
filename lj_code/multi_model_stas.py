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
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']


def stas_accuracy(data_4o, data_o1, path_save):
    total_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    correct_both_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    error_both_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    o1_correct_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    g4o_correct_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    save_results = []
    # 将data_o1转换为以index_sen为key的字典，方便查找
    data_o1_dict = {item.get('index_sen'): item for item in data_o1 if item.get('index_sen') is not None}

    # 读取data_4o中的每一条数据，基于index_sen来获取data_o1中的数据，对比final_judgement结果，并根据error_type判断结果是否正确
    for instance in tqdm(data_4o):
        index_sen = instance.get('index_sen', None)
        final_judgement_4o = instance.get('final_judgement', None)
        error_type_4o = instance.get('error_type', None)
        if index_sen is None or index_sen not in data_o1_dict:
            print(f'index_sen错误或在o1数据中未找到：{index_sen}')
            continue
        corresponding_o1_instance = data_o1_dict[index_sen]
        final_judgement_o1 = corresponding_o1_instance.get('final_judgement', None)
        if error_type_4o not in total_type:
            continue
        total_type[error_type_4o] += 1
        instance['final_judgement_4o'] = final_judgement_4o
        instance['final_judgement_o1'] = final_judgement_o1
        if error_type_4o == 'right':
            if final_judgement_4o == 'correct' and final_judgement_o1 == 'correct':
                correct_both_type[error_type_4o] += 1
            elif final_judgement_4o == 'correct':
                g4o_correct_type[error_type_4o] += 1
                save_results.append(instance)
            elif final_judgement_o1 == 'correct':
                o1_correct_type[error_type_4o] += 1
                save_results.append(instance)
            else:
                error_both_type[error_type_4o] += 1
                save_results.append(instance)
        elif error_type_4o == 'label_error' or error_type_4o == 'boundary_error' or error_type_4o == 'redundant':
            if final_judgement_4o == 'incorrect' and final_judgement_o1 == 'incorrect':
                correct_both_type[error_type_4o] += 1
            elif final_judgement_4o == 'incorrect':
                g4o_correct_type[error_type_4o] += 1
                save_results.append(instance)
            elif final_judgement_o1 == 'incorrect':
                o1_correct_type[error_type_4o] += 1
                save_results.append(instance)
            else:
                error_both_type[error_type_4o] += 1
                save_results.append(instance)
    print(f'整体error_type分布：{total_type}')
    print(f'两种模型均正确判断的分布：{correct_both_type}')
    print(f'仅4o模型正确判断的分布：{g4o_correct_type}')
    print(f'仅o1模型正确判断的分布：{o1_correct_type}')
    print(f'两种模型均错误判断的分布：{error_both_type}')
    json.dump(save_results, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False)    

    

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    data_4o = read_json('llmout_lyh/nw/nw-tc-test-llmsimp-4o-mini.json')
    data_o1 = read_json('llmout_lyh/nw/nw-tc-test-llmsimp.json')
    stas_accuracy(data_4o, data_o1, '4o_vs_o1_nw-tc-test.json')
    """
    dataset = ['test'] #dev, 
    source = ['bn'] #["nw",  "bn", "bc" ]# 'bn'
    target = ['tc'] #['tc', 'bn', 'nw', 'bc'] # 'tc',
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                data = read_json(f'llmout/{i}/{i}-{j}-{k}-llmsimp.json')
                path_llmout = f'llmout/{i}/{i}-{j}-{k}-llmerror.json'
                stas_accuracy(data, path_llmout)
    """

    
