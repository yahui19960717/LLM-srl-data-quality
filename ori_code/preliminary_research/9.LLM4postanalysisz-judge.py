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
    num_right, num_wrong = 0, 0
    num_right_iserror = 0
    count = 0
    dic_error_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    dic_org_error_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    right_span, error_span = set(), set()
    for instance in tqdm(data): # 一个句子
        dic_org_error_type[instance['error_type']]+=1
        try:    
            res =json.loads(instance['response'])
            judge = res['correct']
            # print(instance["error_type"], judge)
            if instance["error_type"]!="right" and instance["error_type"]=="redundant" and (judge==True or judge=="true"):
                print("sentence: ", instance['sentences'])
                print("prd: ",instance['predicate'])
                print("arg: ", instance['selected_span'])
                print("error_type: ", instance['error_type'])
                print("prob: ",instance['predict_prob'])
                print("LLM judge response: ", instance['response'])
                # import pdb;pdb.set_trace()
            if instance["error_type"]=="right" and (judge==True or judge=="true"):
                num_right+=1
                dic_error_type[instance["error_type"]]+=1
                right_span.add(tuple(instance['org_span']))
            elif instance["error_type"]!="right" and (judge == False or judge=="false"):
                num_right+=1
                num_right_iserror += 1
                dic_error_type[instance["error_type"]]+=1
                right_span.add(tuple(instance['org_span']))
            else:
                num_wrong += 1
                # import pdb;pdb.set_trace()
                error_span.add(tuple(instance['org_span']))
            if judge==False or judge=="false":
                results.append(tuple(instance['org_span']))
        except:
            try:
                json_str = json_str = re.sub(r'("reason":\s*")([^"]+)"([^"]*)"', r'\1\2\3"', instance["response"])
                res =json.loads(json_str)
                judge = res['correct']
                # print(instance["error_type"], judge)
                if instance["error_type"]=="right" and (judge==True or judge=="true"):
                    num_right+=1
                    right_span.add(tuple(instance['org_span']))
                    dic_error_type[instance["error_type"]]+=1
                elif instance["error_type"]!="right" and (judge == False or judge=="false"):
                    num_right+=1
                    num_right_iserror += 1
                    dic_error_type[instance["error_type"]]+=1
                    right_span.add(tuple(instance['org_span']))    
                else:
                    
                    num_wrong += 1
                    error_span.add(tuple(instance['org_span']))
                if judge==False or judge=="false":
                    results.append(tuple(instance['org_span']))
            except:
                count+=1
                error_span.add(tuple(instance['org_span']))
                # print(instance['response'])
                # # import pdb;pdb.set_trace()
                # print(count)
    assert (len(right_span)+len(error_span))==len(data)
    assert num_right == len(right_span)
    assert num_wrong+count == len(error_span)
    print(f'原始数据中的错误分布：{dic_org_error_type}')
    print(f'大模型判断正确的spans个数为:{num_right}') 
    print(f'大模型判断错误的spans个数为 {num_wrong}' )
    print(f'大模型判断正确的错误span个数为(判断正确的spans个数的子集）：{num_right_iserror}')
    print(f'大模型输出错误的spans个数为:{count}')
    print(f'判断正确的span中错误分布(说明大模型擅长解决什么方面的问题）：{dic_error_type}')
    print(f'判断正确的span中错误分布：right ratio:{dic_error_type["right"]/dic_org_error_type["right"]*100:.2f}')
    print(f'\t label error ratio: {dic_error_type["label_error"]/dic_org_error_type["label_error"]*100:.2f}')
    print(f'\t boundary_error ratio: {dic_error_type["boundary_error"]/dic_org_error_type["boundary_error"]*100:.2f}')
    print(f'\t  redundant ratio: {dic_error_type["redundant"]/dic_org_error_type["redundant"]*100:.2f}')
    print(f'大模型准确率为:{num_right/(num_right+num_wrong+count) * 100:.3f}')
    return results, right_span, error_span

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
    print("小模型的PRF值：")
    P_or, R_or, F_or = obtain_prf(dic_gold_spans.keys(), dic_pred_spans.keys())
    print("*"*50)
    print("小模型去掉不确定span之后的PRF值：")
    small_model_pred = dic_pred_spans.keys() - selected_small_model
    P_sm, R_sm, F_sm = obtain_prf(dic_gold_spans.keys(), small_model_pred)
    print("*"*50)
    print("大模型从不确定span中召回判断之后的PRF值：")
    selected_llm = set(results)
    llm_pred = dic_pred_spans.keys() - selected_llm
    P_llm, R_llm, F_llm = obtain_prf(dic_gold_spans.keys(), llm_pred)
    print("*"*50)
    print("-"*50)

    import pdb;pdb.set_trace()  


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 
    source = ["nw",  "bn", "bc" ]# 'bn'
    target = [ 'bn', 'tc', 'nw', 'bc']
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                print("--"*100)
                distributions = read(f"prob_distribution/{k}/{i}-{j}-{k}-distribution.pt")
                llmresults = f'../llmout_lyh/{i}/{i}-{j}-{k}-llmsimp.json'
                # llmresults = read_json(f'llmout/llmout_frames/{i}/{i}-{j}-{k}-frames-judge-zeroshotvote.json') 
                selected_spans = read_json(f"selected_spans_unite/{i}/{i}-{j}-{k}.json")
                LLM_result_analysis(llmresults)
                LLM_result_prf(llmresults, distributions, selected_spans)
                print("工作保存完成！")
                print("--"*100)
                exit()
            
    