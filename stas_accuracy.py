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


def stas_accuracy(data, path_save):
    correct_num = 0
    total_num = len(data)
    core_total_num = 0
    core_correct_num = 0
    nocore_total_num = 0
    nocore_correct_num = 0
    old_correct_num = 0
    old_core_correct_num = 0
    old_nocore_correct_num = 0
    err_results = []
    for instance in tqdm(data):
        final_judgement = instance.get('final_judgement', None)
        error_type = instance.get('error_type', None)
        selected_span = instance.get('selected_span', None)
        org_span = instance.get('org_span', None)
        conflict_span = instance.get('conflict_span', None)
        temp_span = selected_span[1].split("-")[-1]
        if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
            core_total_num += 1
        else:
            nocore_total_num += 1        
        if final_judgement == 'correct' and (error_type == 'right' or error_type == 'boundary_error' and selected_span[2][0] == org_span[2] and selected_span[2][1] == org_span[3]):
            correct_num += 1
            if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
                core_correct_num += 1
            else:
                nocore_correct_num += 1
        elif final_judgement == 'incorrect' and error_type != 'right':
            correct_num += 1
            if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
                core_correct_num += 1
            else:
                nocore_correct_num += 1
        else:
            if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
                err_results.append(instance)
        # if error_type == 'right' or error_type == "boundary_error" and selected_span[2][0] == org_span[2] and selected_span[2][1] == org_span[3]:
        if error_type == 'right' or error_type == "boundary_error" and selected_span[2][0] == conflict_span[2] and selected_span[2][1] == conflict_span[3]:
            old_correct_num += 1
            if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
                old_core_correct_num += 1
            else:
                old_nocore_correct_num += 1
    
    json.dump(err_results, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    core_accuracy = core_correct_num / core_total_num if core_total_num > 0 else 0
    nocore_accuracy = nocore_correct_num / nocore_total_num if nocore_total_num > 0 else 0
    accuracy = correct_num / total_num if total_num > 0 else 0
    old_core_accuracy = old_core_correct_num / core_total_num if core_total_num > 0 else 0
    old_nocore_accuracy = old_nocore_correct_num / nocore_total_num if nocore_total_num > 0 else 0
    old_accuracy = old_correct_num / total_num if total_num > 0 else 0
    print(f"Total Instances: {total_num}, Correct Instances: {correct_num}, Accuracy: {accuracy:.4f}, Previous Accuracy: {old_accuracy:.4f}")
    print(f"Core Role Instances: {core_total_num}, Correct Core Role Instances: {core_correct_num}, Core Role Accuracy: {core_accuracy:.4f}, Previous Core Role Accuracy: {old_core_accuracy:.4f}")
    print(f"Non-Core Role Instances: {nocore_total_num}, Correct Non-Core Role Instances: {nocore_correct_num}, Non-Core Role Accuracy: {nocore_accuracy:.4f}, Previous Non-Core Role Accuracy: {old_nocore_accuracy:.4f}")
    return accuracy, core_accuracy, nocore_accuracy


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
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

    
