# 构建prompt来应用LLM

import os
from tqdm import tqdm
import json
import re, torch
from collections import defaultdict, Counter
from torch.mps import set_per_process_memory_fraction
from config import ARG, ARGM
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']

def read(file,device="cuda"):
    data = torch.load(file,map_location=device,mmap=True, weights_only=True)
    return data

def write_json(data, file):
    with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=0)
# 读json文件 index
def read_json(file):
  with open(file, "r", encoding="utf-8") as f:
      data = json.load(f)
  return data 

def obtain_prf(gold, pred):
    correct = len(gold & pred)
    total_pred = len(pred)
    total_gold = len(gold)
    P = correct / (total_pred + 1e-12)
    R = correct / (total_gold + 1e-12)
    F = (2 * P * R ) / (P + R + 1e-12)
    print(f"正确的spans：{correct}, Predict spans数：{total_pred}, Gold spans数：{total_gold}")
    # import pdb;pdb.set_trace()
    print(f'corr. {correct}, excess: {total_pred-correct}, miss: {total_gold-correct}')
    print(f'{correct} {total_pred-correct} {total_gold-correct}')
    print(f'P: {P*100:.2f}, R: {R*100:.2f}, F: {F*100:.2f}')
    print(f'{P*100:.2f} {R*100:.2f} {F*100:.2f}')
    return P, R, F

def analysis_llm_res(data):
    '''
    dict_keys(['index_sen', 'sentences', 
    'selected_span', 'predicate', 'selected_other_spans', 
    'predict_prob', 'error_type', 'gold_label', 'response'])
    '''
    # 先检查下大模型对错率
    results=[]
    num_right_but_wrong = 0
    span = []
    dic_error_type = {"yesno":0,  "nono":0, "noyes":0, "yesyes":0}
    for instance in tqdm(data): # 一个句子
        judge = instance['final_judgement']
        if judge != "correct" and instance["error_type"]=="right":
            num_right_but_wrong += 1
            span.append(instance)
            if instance['argument_extraction_evaluation']=="yes" and instance['role_evaluation']=="no":
                dic_error_type["yesno"]+=1
            elif instance['argument_extraction_evaluation']=="no" and instance['role_evaluation']=="no":
                dic_error_type["nono"]+=1
            elif instance['argument_extraction_evaluation']=="no" and instance['role_evaluation']=="yes":
                dic_error_type["noyes"]+=1
            elif instance['argument_extraction_evaluation']=="yes" and instance['role_evaluation']=="yes":
                dic_error_type["yesyes"]+=1
            else:
                import pdb;pdb.set_trace()
            print("----------------------"*2)
            print("sentence:", instance['sentences'])
            print("predicate: " , instance['predicate'])
            print("span:", instance['selected_span'][0], ":", instance['selected_span'][1]) 
            print("'argument_extraction_evaluation':", instance['argument_extraction_evaluation'])
            print("role_evaluation: ", instance['role_evaluation'])
            print("----------------------"*2)
            print("\n")
            # import pdb;pdb.set_trace()
            
          
    print(f'The number of spans:{num_right_but_wrong}') 
    print(dic_error_type)
    import pdb;pdb.set_trace()
    return span

def judge_label(label):
    if label.split("-")[0]=="C":
        label = "-".join(label.split("-")[1:])
    elif label.split("-")[0]=="R":
        label = "-".join(label.split("-")[1:])
    return label

def get_label_distribution(right):
    right_label = {"core":0, "noncore":0}
    right_dic = defaultdict(int)
    for span in right:
        label = conll12_label[span[4]]
        label = judge_label(label)
        right_dic[label]+=1
        if label in ARG.keys():
            right_label['core']+=1
        else:
            right_label['noncore']+=1
            assert label in ARGM.keys()

    assert right_label['core']+right_label['noncore']==len(right)
    sorted_items = sorted(right_dic.items(), key=lambda x: x[1])
    return right_label, sorted_items

def LLM_result_analysis(data):
    _, right, error= analysis_llm_res(data)


    right_label, right_dic = get_label_distribution(right)
    error_label, error_dic = get_label_distribution(error)
    print(f"Right: {right_label} \n {right_dic} ")
    print(f"Error: {error_label} \n {error_dic} ")

     
def LLM_result_prf(data, all_data, selected_spans):
    ## 看下预测结果中的PRF值，看下小模型选择之后的PRF值，看下大模型选择之后的PRF值
    results, _, _ = analysis_llm_res(data)
    all_gold_spans = all_data["all_gold_spans"]
    dic_pred_spans = all_data['dic_pred_spans']
    selected_small_model = set([tuple(key) for key in selected_spans['selected_spans']])
    dic_gold_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_gold_spans}
    print("-"*50)
    print("*"*50)
    print("SML's PRF1：")
    P_or, R_or, F_or = obtain_prf(dic_gold_spans.keys(), dic_pred_spans.keys())
    print("*"*50)
    print("(SML - uncertain spans)'s PRF：")
    small_model_pred = dic_pred_spans.keys() - selected_small_model
    P_sm, R_sm, F_sm = obtain_prf(dic_gold_spans.keys(), small_model_pred)
    print("*"*50)
    print("LLM (after recall some spans) PRF：")
    selected_llm = set(results)
    llm_pred = dic_pred_spans.keys() - selected_llm
    P_llm, R_llm, F_llm = obtain_prf(dic_gold_spans.keys(), llm_pred)
    print("*"*50)
    print("-"*50)

    import pdb;pdb.set_trace()  


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 
    source = ["bn"]# 'bn'
    target = ["tc"]
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                print("--"*100)
                distributions = read(f"../prob_distribution/{k}/{i}-{j}-{k}-distribution.pt")
                llmresults = read_json(f'../llmout_lyh/{i}/{i}-{j}-{k}-llmsimp-4o-mini.json')
                selected_spans = read_json(f"../selected_spans_unite/{i}/{i}-{j}-{k}.json")
                LLM_result_analysis(llmresults)
                exit()
            
    