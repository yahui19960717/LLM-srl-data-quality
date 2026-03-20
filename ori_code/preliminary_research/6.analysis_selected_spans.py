# 在dev上获得 不同方法在不确定性得分的阈值, 根据错误span的PR F来获得阈值

import os
from matplotlib import legend
import torch
import json
from torch.mps import set_per_process_memory_fraction
from collections import defaultdict
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']

def read(file,device="cuda"):
    data = torch.load(file,map_location=device,mmap=True, weights_only=True)
    return data

# 写json文件 
def write_json(data, file):
    with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=0)
# 读json文件 index
def read_json(file):
  with open(file, "r", encoding="utf-8") as f:
      data = json.load(f)
  return data

# 判断两个span是否有交集
def has_overlap(span1, span2):
    # 判断span1和span2是否有交集
    return span1[0] <= span2[1] and span2[0] <= span1[1]

def obtain_prf(gold, pred):
    correct = len(gold & pred)
    total_pred = len(pred)
    total_gold = len(gold)
    P = correct / (total_pred + 1e-12)
    R = correct / (total_gold + 1e-12)
    F = (2 * P * R ) / (P + R + 1e-12)
    print(f"正确的spans：{correct}, Predict spans数：{total_pred}, Gold spans数：{total_gold}")
    # import pdb;pdb.set_trace()
    # print(f'corr. {correct}, excess: {total_pred-correct}, miss: {total_gold-correct}')
    # print(f'{correct} {total_pred-correct} {total_gold-correct}')
    # print(f'P: {P*100:.2f}, R: {R*100:.2f}, F: {F*100:.2f}')
    # print(f'{P*100:.2f} {R*100:.2f} {F*100:.2f}')
    return P, R, F

def categorize_error(gold, pred):
    errors= {'label_error': 0, "boundary_error": 0,  "redundant": 0}
    for key in pred:
        flag = 0
        sen_id, prd, start, end, label = key
        temp_span = (sen_id, prd, start, end)
        # 标签错误
        if temp_span in gold and gold[temp_span]!=label:
            errors['label_error']+=1
        else:
            flag = any(
                g[0] == sen_id and g[1] == prd and has_overlap([start, end], [g[2], g[3]]) 
                for g in gold
            )
            # 交叉
            if flag == True:
                errors["boundary_error"]+=1
            # 过度标注错误：如果预测标签的span不在真实标签中
            if flag == False and temp_span not in gold:
                errors["redundant"]+=1
        
    return errors

            


def analysis_selected_spans(data, path_save):
    merged_spans = read_json(path_save)
    all_gold_spans = data['all_gold_spans']
    dic_gold_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_gold_spans}
    dic_gold_spans_span_label = {(sen_id, prd, start, end): label for sen_id, prd, start, end, label, score in all_gold_spans}
    dic_pred_spans = data['dic_pred_spans']
    dic_predicate_probs = defaultdict(list)

    # 将错误的spans保存起来
    right_pred_spans = []
    error_pred_spans = []
    for key in dic_pred_spans.keys():
        
        if key in dic_gold_spans.keys():
            right_pred_spans.append(key)
        else:
            error_pred_spans.append(key)
        temp = dic_pred_spans[key]
        dic_predicate_probs[" ".join([str(key[0]),str(key[1])])].append(max(temp))

    right_error_span = [key for key in merged_spans['selected_spans']  if tuple(key) in set(error_pred_spans)] 
    # print("*"*50)
    # print(f"{'全部找出的候选错误spans': <20} {len(merged_spans['selected_spans'])} \n {'候选中正确的span数': <20} {len(merged_spans['selected_spans'])-len(right_error_span)}\n {'真正错误的spans':<20} {len(right_error_span)}")
    # print("-"*10)
    # 分析错误类型
    errors = categorize_error(dic_gold_spans_span_label, right_error_span)
    # print(errors)
    # print("*"*50)
    selected_spans = [key for key in merged_spans['selected_spans'] ]

    
    selected_llm = set(error_pred_spans)
    dic_pred_selected = dic_pred_spans.keys()-selected_llm
    print("-"*60)
    # print("原始的PRF：")
    # P,R,F = obtain_prf(dic_gold_spans.keys(),dic_pred_spans.keys())
    # print("-"*20)
    # print("正确的PRF：")
    # P1,R1,F1=obtain_prf(dic_gold_spans.keys(), dic_pred_selected)
    # print(f'{P*100:.2f},{R*100:.2f},{F*100:.2f}')
    # print(f'{P1*100:.2f},{R1*100:.2f},{F1*100:.2f}')
    # print(f'{(F1-F)*100:.2f}')
    # print("-"*60)
    print(f"正确的:{len(right_error_span)}, 错误的：{len(selected_spans)-len(right_error_span)}")
    print(f"删掉的和总共的:{len(selected_spans)}, {len(dic_pred_spans)}")
    print(f"删掉的占比:{len(selected_spans)/len(dic_pred_spans)}")
    print("-"*60)

    
    # 分析错误标签的分布

    # import pdb;pdb.set_trace()


    
    

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 
    source = ["nw",  "bn", "bc" ]# 'bn'
    target = [ 'tc', 'bn', 'nw', 'bc'] #,
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                distributions = read(f"prob_distribution/{k}/{i}-{j}-{k}-distribution.pt")
                # path_save = f"selected_spans_merge/{i}/{i}-{j}-{k}.json"
                path_save = f"selected_spans_unite/{i}/{i}-{j}-{k}.json"
                analysis_selected_spans(distributions,  path_save)
                # import pdb;pdb.set_trace()


    print("工作保存完成！")