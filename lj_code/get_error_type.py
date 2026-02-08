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


def stas_accuracy(data, type, path_save):
    data_list = []
    # 按error type选择指定类型的数据，保存在data_list中
    for instance in tqdm(data):
        index_sen = instance.get('index_sen', None)
        sentence = instance.get('sentences', None)
        predicate = instance.get('predicate', None)
        select_span = instance.get('selected_span', None)
        final_judgement_ds = instance.get('final_judgement', None)
        final_judgement_o1 = instance.get('final_judgement_o1', None)
        final_judgement_g4o = instance.get('final_judgement_4o', None)
        error_type = instance.get('error_type', None)

        if error_type == type and final_judgement_ds == 'correct':
            data_list.append(instance)
            print(f'{index_sen} \t {sentence} \t {predicate} \t {select_span[0]} \t {select_span[1]} \t {error_type} \t {final_judgement_ds} \t {final_judgement_o1} \t {final_judgement_g4o}')
        
    json.dump(data_list, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False)    

    

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    data = read_json('llmout_ds/4o_vs_o1_nw-tc-test-ds-simp.json')
    stas_accuracy(data, "label_error", 'llmout_ds/4o_vs_o1_nw-tc-test-ds-label.json')
    
