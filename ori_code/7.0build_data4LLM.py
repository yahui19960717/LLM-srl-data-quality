# 创建LLM需要的数据
import os
import torch
from torch.mps import set_per_process_memory_fraction
import json
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

def obtain_error_type(selected_data, error_pred_spans, gold):
    '''对于模型自己选择出来的全部不确定spans进行错误类型的判断'''
    # 每个span是对的还是错的，如果是错误的是什么类型的错误
    # Error types include： right, label_error, boundary_error, redundant
    dic_error_type = {} 
    # Record error spans of selected spans
    selected_error_spans = []
    # Record label_error's right label
    dic_rightlabel_of_errorlabelspans = {}
    # Record boundary conflict 
    dic_conflict_of_boundaryerrorspans = {}


    # selected spans divided into right and wrong
    for key in selected_data['selected_spans']:
        key = tuple(key)
        if tuple(key) not in set(error_pred_spans):
            dic_error_type[key] = "right"
        else:
            selected_error_spans.append(key)

    # the wrong spans devided into three types
    for key in selected_error_spans:
        flag = 0
        key = tuple(key)
        sen_id, prd, start, end, label = key
        temp_span = (sen_id, prd, start, end)
        # Label Error
        if temp_span in gold and gold[temp_span]!=label:
            dic_error_type[key] = "label_error"
            dic_rightlabel_of_errorlabelspans[key] = gold[temp_span]
        else:
            for g in gold:
                if g[0] == sen_id and g[1] == prd and has_overlap([start, end-1], [g[2], g[3]-1]) and gold[g]==label:
                    # import pdb;pdb.set_trace()
                    overlap_entity = [g[2], g[3]]  # 或直接保存 g
                    dic_conflict_of_boundaryerrorspans[key] = [sen_id, prd,g[2], g[3], label]
                    flag = True
                    break 
            # Boundary Error
            if flag == True:
                dic_error_type[key] = "boundary_error"
            # Over-annotate,i.e., redundant
            else:
                if flag == False and temp_span not in gold:
                    dic_error_type[key] = "redundant"
                else:
                    import pdb;pdb.set_trace()
                    print("WARNING!ERROR!")
                
    assert len(dic_error_type)==len(selected_data['selected_spans'])
    return dic_error_type, dic_conflict_of_boundaryerrorspans, dic_rightlabel_of_errorlabelspans

def select_spans(data, index_data, merged_data, path_save):
    '''主要是用来'''
    all_gold_spans = data['all_gold_spans'] # gold spans
    # gold spans dict: (sen_id, prd, span,label):score
    dic_gold_spans = {(sen_id, prd, start, end, label): score for sen_id, prd, start, end, label, score in all_gold_spans} 
    # (prd, span):  label
    gold = {(sen_id, prd, start, end): label for sen_id, prd, start, end, label, score in all_gold_spans} 
    # pred spans  
    dic_pred_spans = data['dic_pred_spans']
    # (sen_id, prd)为key存放对应的标注，i.e.,prd-level的标注
    dic_prdlevel_predanno = defaultdict(list)

    # Obtain right and error pred spans
    right_pred_spans = []
    error_pred_spans = []
    for key in dic_pred_spans.keys():
        if key in dic_gold_spans.keys():
            right_pred_spans.append(key)
        else:
            error_pred_spans.append(key)

        dic_prdlevel_predanno[(key[0],key[1])].append(key)


    # Obtain selected uncertainty spans
    selected_data = read_json(merged_data) 

    # Obtain error types
    dic_error_type, conflict, right_label = obtain_error_type(selected_data, error_pred_spans, gold)

    
    span_level_results = [] # input LLM 
    index_sen = 0
    for key in selected_data['selected_spans']: # key 可能是正确的也可能是错误的
        dic_span_level = {}
        # sen[0]:sentence sen[1]:pred; sen[2]:gold
        sen_id, prd, start, end, pred_label = tuple(key) 
        sen = index_data[str(sen_id)] # 句子-pred results
        dic_span_level['index_sen'] = index_sen
        dic_span_level['sentences']=" ".join(sen[0])
        other_spans = dic_prdlevel_predanno[(sen_id, prd)] #同一个句子的spans
        span_labels = []
        for span in other_spans: # 遍历其他的spans
            if tuple(key) != span:
                span_labels.append(":".join([" ".join(sen[0][span[2]:span[3]]), conll12_label[span[4]]])) 
            else:
                key_span = " ".join(sen[0][span[2]:span[3]])
                dic_span_level['selected_span'] = [key_span, conll12_label[span[4]], [span[2], span[3]]]
                predicate = sen[0][span[1]-1]
        dic_span_level['predicate'] = predicate 
        dic_span_level['org_span'] = key
        dic_span_level['selected_other_spans'] = span_labels # 预测的其他spans
        dic_span_level['predict_prob'] = max(dic_pred_spans[tuple(key)])
        dic_span_level['error_type']= dic_error_type[tuple(key)]
        
        if dic_error_type[tuple(key)]== "label_error": #标签错误的情况下
            dic_span_level["gold_label"] = conll12_label[gold[(sen_id, prd, start, end)]]
        else:
            dic_span_level["gold_label"]=None # 标签不错的情况也是None
        if dic_error_type[tuple(key)]=="boundary_error":
            dic_span_level['conflict_span'] = conflict[tuple(key)] # 重叠的span
        else:
            dic_span_level['conflict_span'] = None # 预测结果
        span_level_results.append(dic_span_level)
        index_sen += 1
      
    assert index_sen == len(span_level_results)
    write_json(span_level_results, path_save)

    
    

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 
    source = ["nw", "bn", "bc"  ]#  nw-tc完成, "nw", 
    target = ['tc', 'bn', 'nw', 'bc']
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')  
                # Obtain gold/pred probabilities      
                distributions = read(f"/data/yhliu/Span-based-SRL/20251013/prob_distribution/{k}/{i}-{j}-{k}-distribution.pt") # 概率分布
                # selected_span_path = f"selected_spans/{i}/{i}-{j}-{k}.json"
                selected_span_path = f"../selected_spans_unite/{i}/{i}-{j}-{k}.json"
                index_data = read_json(f"../index_sen/{i}/{j}.{k}.json") # 句子-pred results
                path_save = f'../forllm_unite_newest/{i}/{i}-{j}-{k}.json'
                select_spans(distributions, index_data, selected_span_path, path_save)
                
            
    print("工作保存完成！")