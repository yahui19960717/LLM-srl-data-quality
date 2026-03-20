## 对候选集合进行判别，效果如何，先高优进行核心角色判别，看初步效果

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

def parse_response(instance, response):
    try:
        parse_tag = False
        res_str = response.strip()
        if not res_str:
            print("Error:empty response!")
            return response, parse_tag
        if res_str.startswith("```json"):
            res_str = res_str[7:]
        if res_str.endswith("```"):
            res_str = res_str[:-3]
        res_str = res_str.strip()

        res_data = json.loads(res_str)
        parse_tag = True
        res_data['index_sen'] = instance['index_sen']
        res_data['sentences'] = instance['sentences']
        res_data['predicate'] = instance['predicate']
        res_data['selected_span'] = instance['selected_span']
        res_data['error_type'] = instance['error_type']
        res_data['gold_label'] = instance['gold_label']
        res_data['org_span'] = instance['org_span']
        res_data['conflict_span'] = instance['conflict_span']
        return res_data, parse_tag
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag
    except Exception as e:
        print(f"其他错误: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag


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
    count = 0
    # 先检查下大模型对错率
    results=[]
    num_right, num_wrong = 0, 0
    num_right_iserror = 0 # 错误span被LLM正确判断出来了
    rule_num_error, rule_num_right = 0, 0
    parse_num_error = 0
    empty_num = 0

    dic_error_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0} #llm: parse right
    dic_org_error_type = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    right_span, error_span = set(), set() # LLM的结果
    parse_error = set()
    rule_right_span, rule_error_span = set(), set() # rule的结果
    all_right_span, all_error_span = set(), set()
    # dic_error_type_all = {"right":0,  "label_error":0, "boundary_error":0, "redundant":0}
    
    results_right_llm = []
    
    for instance in tqdm(data): # 一个句子
        dic_org_error_type[instance['error_type']]+=1
        res = instance['response']
        res_data, res_tag = parse_response(instance, res) # LLM 判断出来的结果
        
        try:
            if res_tag == False and (res_data==None or type(res_data)==str): # LLM的输出解析错误，我们将这种视为大模型判断错误了
                if res_data==None:
                    empty_num+=1
                parse_num_error += 1
                parse_error.add(tuple(instance['org_span']))
                all_error_span.add(tuple(instance['org_span']))
                results.append(tuple(instance['org_span']))
                print(res_data)
            else:
                if "correct" in res_data.keys(): # 根据规则判断出来的
                    judge = instance['response']['correct'] 
                    if judge == False and instance['error_type']!="right": # 错错
                        rule_num_right+=1
                        rule_right_span.add(tuple(instance['org_span']))
                        all_right_span.add(tuple(instance['org_span']))
                        results_right_llm.append(tuple(instance['org_span']))
                    elif judge == True and instance['error_type']=="right": # 对对
                        rule_num_right += 1
                        rule_right_span.add(tuple(instance['org_span']))
                        all_right_span.add(tuple(instance['org_span']))
                        results_right_llm.append(tuple(instance['org_span']))
                    else:
                        rule_num_error += 1
                        rule_error_span.add(tuple(instance['org_span']))
                        all_error_span.add(tuple(instance['org_span']))
                        results.append(tuple(instance['org_span']))
                else:# LLM解析正确的：
                    assert res_tag == True
                    judge = res_data['final_judgement']
                    if judge == "correct" and instance["error_type"]=="right":
                        num_right += 1
                        dic_error_type[instance["error_type"]]+=1
                        right_span.add(tuple(instance['org_span']))
                        all_right_span.add(tuple(instance['org_span']))
                    elif judge == "incorrect" and instance['error_type']!="right":
                        num_right+=1
                        num_right_iserror += 1
                        dic_error_type[instance["error_type"]]+=1
                        right_span.add(tuple(instance['org_span']))
                        all_right_span.add(tuple(instance['org_span']))
                    else:
                        num_wrong += 1
                        error_span.add(tuple(instance['org_span']))
                        all_error_span.add(tuple(instance['org_span']))
                        if judge == "incorrect" and instance['error_type']=="right":
                            count += 1
                    if judge=="incorrect": # 大模型判断span为错误的结果
                        results.append(tuple(instance['org_span']))
                    if judge=="correct":
                        results_right_llm.append(tuple(instance['org_span']))
        except:
            import pdb;pdb.set_trace
            print(11)
    # import pdb;pdb.set_trace()
    assert (len(right_span)+len(error_span))==(len(data)-parse_num_error-rule_num_error-rule_num_right)
    assert num_right == len(right_span)
    assert num_wrong == len(error_span)
    assert len(results) == (count + num_right_iserror + parse_num_error + rule_num_error)
    print(f"*****************Number info*************************")
    print(f'[Data number]: {len(data)}')
    print(f'[Rule judege]--right number: {rule_num_right}, error number: {rule_num_error}')
    print(f'[Parse Error]--error number: {parse_num_error}, where empty number is {empty_num}')
    print(f'[LLM judge]--: right number: {num_right}, error number: {num_wrong}, right in error spans: {num_right_iserror}')
    # print(f'The number of spans judged correctly by LLM:{num_right}') 
    # print(f'The number of spans judgedd incorrectly by LLM: {num_wrong}' )
    # print(f'The number of error spans judeged correctly by LLM: {num_right_iserror}')
    print(f'****************Error type infor*************')
    print(f'原始数据中的错误分布：{dic_org_error_type}')
    print(f'判断正确的span中错误分布(说明大模型擅长解决什么方面的问题）：{dic_error_type}')
    print(f'判断正确的span中错误分布：right ratio:{dic_error_type["right"]/dic_org_error_type["right"]*100:.2f}')
    print(f'\t label error ratio: {dic_error_type["label_error"]/dic_org_error_type["label_error"]*100:.2f}')
    print(f'\t boundary_error ratio: {dic_error_type["boundary_error"]/dic_org_error_type["boundary_error"]*100:.2f}')
    print(f'\t  redundant ratio: {dic_error_type["redundant"]/dic_org_error_type["redundant"]*100:.2f}')
    print(f'LLM accuracy:{num_right/(num_right+num_wrong) * 100:.3f}')
    # import pdb;pdb.set_trace()
    return results_right_llm, results, all_right_span, all_error_span

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


    # right_label, right_dic = get_label_distribution(right)
    # error_label, error_dic = get_label_distribution(error)
    # print(f"Right: {right_label} \n {right_dic} ")
    # print(f"Error: {error_label} \n {error_dic} ")
   
def LLM_result_prf(data, all_data, selected_spans):
    ## 看下预测结果中的PRF值，看下小模型选择之后的PRF值，看下大模型选择之后的PRF值
    print(f"The number of data is : {len(data)}")
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

    # import pdb;pdb.set_trace()  






def Select_add_spans(all_data):
    
    all_gold_spans = all_data["all_gold_spans"]
    all_pred_spans = all_data['all_pred_spans']
    all_candidate_spans = all_data['all_candidate_spans']

    
    dic_gold_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_gold_spans}
    dic_pred_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_pred_spans}
    # 已经去掉了NULL和PAD
    dic_cand_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_candidate_spans if label !=0 and label !=1}
    # dic_cand_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_candidate_spans}

    
    # 去掉预测结果中的，
    print(f'Original number of candidate spans :{len(dic_cand_spans)}')
    print(f'Predict spans number is {len(dic_pred_spans)}')
    cand_spans_set = dic_cand_spans.keys() - dic_pred_spans.keys()
    print(f'Different spans of canddiate spans and predict spans: {len(cand_spans_set)}')

    dic_candidate_spans_core = {}
    dic_candidate_spans_noncore = {}
    right_candidate, right_candidate_core, right_candidate_noncore = {}, {}, {}
    threshold = []
    all_threshold = []
    all_right__threshold = []

    for element in cand_spans_set:
        all_threshold.append(dic_cand_spans[element])
        if conll12_label[element[4]] in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA']: # 若是核心论元
            dic_candidate_spans_core[(element[0], element[1],element[2], element[3], element[4])] = dic_cand_spans[element]
            if element in dic_gold_spans.keys():
                right_candidate_core[(element[0], element[1],element[2], element[3], element[4])] = dic_cand_spans[element]
                threshold.append(dic_cand_spans[element])
        else:
            dic_candidate_spans_noncore[(element[0], element[1],element[2], element[3], element[4])] = dic_cand_spans[element]
            if element in dic_gold_spans.keys():
                right_candidate_noncore[(element[0], element[1],element[2], element[3], element[4])] = dic_cand_spans[element]
        
        if element in dic_gold_spans.keys():
            right_candidate[(element[0], element[1],element[2], element[3], element[4])] = dic_cand_spans[element]
            all_right__threshold.append(dic_cand_spans[element])
        
            
            
        
    
    print(f"The ratio of selected core spansin all candidate spans: {len(dic_candidate_spans_core)} / {len(cand_spans_set)} = {len(dic_candidate_spans_core)/len(cand_spans_set)*100}%")
    print(f'Right spans in candidate set: {len(right_candidate)}')
    print(f'Right core spans in candidate set: {len(right_candidate_core)}')
    print(min(threshold) , max(threshold))
    print(min(all_right__threshold), max(all_right__threshold))
    print(min(all_threshold), max(all_threshold))
    print(f'Right none core spans in candidate set: {len(right_candidate_noncore)}')

    return right_candidate_core, right_candidate_noncore



def stastic_all(all_data):
    all_gold_spans = all_data["all_gold_spans"]
    all_pred_spans = all_data['all_pred_spans']
    all_candidate_spans = all_data['all_candidate_spans']

    
    dic_gold_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_gold_spans}
    dic_pred_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_pred_spans}
    # 已经去掉了NULL和PAD
    dic_cand_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_candidate_spans if label !=0 and label !=1}
    # dic_cand_spans = {(sen_id, prd, start, end, label):score  for sen_id, prd, start, end, label, score in all_candidate_spans}

    # gold :
    dic_gold_spans_core, dic_gold_spans_noncore = {}, {}
    for element in dic_gold_spans:
        # all_threshold.append(dic_cand_spans[element])
        if conll12_label[element[4]] in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA']: # 若是核心论元
            dic_gold_spans_core[(element[0], element[1],element[2], element[3], element[4])] = dic_gold_spans[element]
        else:
            dic_gold_spans_noncore[(element[0], element[1],element[2], element[3], element[4])] = dic_gold_spans[element]
        
    # pred :
    dic_pred_spans_core, dic_pred_spans_noncore = {}, {}
    for element in dic_pred_spans:
        # all_threshold.append(dic_cand_spans[element])
        if conll12_label[element[4]] in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA']: # 若是核心论元
            if element in dic_gold_spans.keys():
                dic_pred_spans_core[(element[0], element[1],element[2], element[3], element[4])] = dic_pred_spans[element]
        else:
            if element in dic_gold_spans.keys():
                dic_pred_spans_noncore[(element[0], element[1],element[2], element[3], element[4])] = dic_pred_spans[element]
        
    
    right_core, right_non = Select_add_spans(all_data)
    print(f"gold:  {len(dic_gold_spans_core)}, {len(dic_gold_spans_noncore)}")  
    print(f"pred:  {len(dic_pred_spans_core)}, {len(dic_pred_spans_noncore)}") 
    print(f"candidate:  {len(right_core)}, {len(right_non)}") 
    # all_right = len(dic_pred_spans_core)+len(right_core)
    # all_pred = len(dic_pred_spans)+len(right_core)
    all_right = len(dic_pred_spans_core)
    all_pred = len(dic_pred_spans)
    all_gold = len(dic_gold_spans)
    P = (all_right)/all_pred
    R = (all_right)/all_gold
    F = 2*P*R/(P+R)
    print(f'{len(dic_pred_spans_core)+len(right_core)}')
    print(f'F: {F*100:.2f}, P: {P*100:.2f}, R: {R*100:.2f}')
    import pdb;pdb.set_trace()
    # 正确的spans：3074, Predict spans数：3891, Gold spans数：4112


    

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 
    source = ["nw"]# 'bn'
    target = ["tc"]
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                print("--"*100) 
                # distributions = read(f"../data-pt/{k}_allspans/{i}-{j}-{k}-allspans.pt")
                distributions = read(f"../prob_distribution/{i}/{i}-{j}-{k}-distribution.pt")
                # json_add = 
                # Select_add_spans(distributions)
                stastic_all(distributions)
                print("工作保存完成！")
                print("--"*100)
                exit()
            
    