import os
import json
import pickle
from collections import defaultdict

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

# 转换data为dict，key为(sentence, prd_word, prd_idx, label)
def transform_golddata_todict(org_data):
    org_dic = {}
    org = org_data.get("annotations", [])
    for i in range(len(org)):
        sen = org[i]['sentence']
        prd_word = org[i]['prd_word']
        prd_idx = org[i]['prd_idx']
        label = org[i]['label']
        key = '\t'.join([sen, prd_word, str(prd_idx), label])
        if key not in org_dic.keys():
            org_dic[key] = org[i]
        else:
            print("org error! have repeat")
            #import pdb;pdb.set_trace()
    #print(f"org data argument num: {len(org_dic)}")
    return org_dic

# 定义函数filter_data，输入has_anno_data和to_anno_data, 输出在to_anno_data中但不在has_anno_data中的数据
def filter_data(has_anno_data, to_anno_data):
    filtered_data = []
    for i in range(len(to_anno_data)):
        key = '\t'.join([to_anno_data[i]['sen'], to_anno_data[i]['prd_word'], str(to_anno_data[i]['prd_idx']), to_anno_data[i]['label']])
        if key not in has_anno_data.keys():
            filtered_data.append(to_anno_data[i])
    
    return filtered_data

if __name__ == "__main__":
    domain = "tc"
    
    # step 1: 读取已经标注过的数据，作为dict
    has_anno_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/anno/tc/annotation_{domain}_smallmodel_removerepeate_322_wlj_final.json")
    has_anno_dict = transform_golddata_todict(has_anno_data)
    
    # step2：读取待标注的数据，然后与has_anno_dict对比过滤，选择不在has_anno_dict中的数据
    to_anno_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/llm/tc/correct_data_{domain}_smnotrecall.json")
    filtered_data = filter_data(has_anno_dict, to_anno_data)
    write_json(filtered_data, f"/data/ljwang/span-SRL-LLM/ori_code/annotation/llm/tc/correct_data_{domain}_smnotrecall_filter.json")
    print(f"filter data num: {len(filtered_data)}")
    