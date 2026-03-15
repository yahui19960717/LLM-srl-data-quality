#!/usr/bin/env python3

import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import os

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

"""
step2：
llm/correct_data_bn_gold.json的文件格式：含sen, prd_word, prd_lemma, prd_idx, label和span_idx；针对每一条数据，以sen、prd_word、prd_idx、label为key；
anno/annotations_gold_o1right_random_wlj_final.json的文件格式：数据在annotations下，含sentence, prd_word, prd_idx, label和selected_spans；针对每一条数据，以sentence、prd_word、prd_idx和label为key；
对比第一个文件中的span_idx和第二个文件中的selected_spans，如果span_idx完全包含在selected_spans中，则作为匹配上，否则是不匹配，匹配和不匹配分别保存及计数；
打印匹配和不匹配的数量；
"""

def compare_json(file_source, file_fix, file_match, file_nomatch, file_partmatch):
    match_list = []
    nomatch_list = []
    partmatch_list = []

    source_dict = {}
    for item_source in file_source:
        sen = item_source["sen"]
        prd_word = item_source["prd_word"]
        prd_idx = item_source["prd_idx"]
        label = item_source["label"]
        span_idx = item_source["span_idx"]
        key_source = (sen, prd_word, prd_idx, label)
        source_dict[key_source] = span_idx
    # 读取第二个文件file_fix中的信息，以sentence、prd_word、prd_idx和label为key,在source_dict中查找span_idx
    for item_fix in file_fix["annotations"]:
        sen = item_fix["sentence"]
        prd_word = item_fix["prd_word"]
        prd_idx = item_fix["prd_idx"]
        label = item_fix["label"]
        selected_spans = item_fix["selected_spans"]
        key_fix = (sen, prd_word, prd_idx, label)
        span_idx = source_dict.get(key_fix, None)
        # 使用span_idx[0]和span_idx[1]-1,依次与selected_spans中的每一个元素对比，对比这里的start和end字段，如果都匹配上，记录匹配加一，否则记录不匹配加1
        if span_idx is None and selected_spans is not None:
            item_fix["span_idx"] = None
            nomatch_list.append(item_fix)
        elif span_idx is not None and selected_spans is None:
            item_fix["span_idx"] = span_idx
            nomatch_list.append(item_fix)
        elif span_idx is None and selected_spans is None:
            match_list.append(item_fix)
        else:
            item_fix["span_idx"] = span_idx
            match_flag = False
            for span in selected_spans:
                if span_idx[0] == span["start"] and span_idx[1]-1 == span["end"]:
                    match_flag = True
                    break
            if match_flag:
                if len(selected_spans) == 1:
                    match_list.append(item_fix)
                else:
                    partmatch_list.append(item_fix)
            else:
                nomatch_list.append(item_fix)
    # 打印匹配和不匹配的数量
    print(f"匹配数量: {len(match_list)}")
    print(f"部分匹配数量: {len(partmatch_list)}")
    print(f"不匹配数量: {len(nomatch_list)}")
    # 保存匹配和不匹配的结果到文件
    write_json(match_list, file_match)
    write_json(nomatch_list, file_nomatch)
    write_json(partmatch_list, file_partmatch)
    
    return 

# step1: 读取两个json文件，
if __name__ == "__main__":
    #调用read_json读取文件：第一个文件是llm/correct_data_bn_gold.json, 第二个文件是annotated_final/annotations_gold_o1right_random_wlj_final.json 
    # o1判断为正确的，随机抽取30，人修改的情况占比
    #file_source = read_json(f"llm/correct_data_bn_gold.json")
    #file_fix = read_json(f"annotated_final/annotations_gold_o1right_random_wlj_final.json")
    #output_match = os.path.join("analysis/", "bn_gold_o1right_random_match.json")
    #output_nomatch = os.path.join("analysis/", "bn_gold_o1right_random_nomatch.json")
    #output_partmatch = os.path.join("analysis/", "bn_gold_o1right_random_partmatch.json")
    #compare_json(file_source, file_fix, output_match, output_nomatch, output_partmatch)
    
    #小模型产出结果，两个大模型都判断为错误的，人随机抽取30，看一下对比情况
    #file_source = read_json(f"llm/incorrect_data_bn_smallmodel_deepseek.json")
    #file_fix = read_json(f"annotated_final/annotations_smallmodel_botherror_random_wlj_final.json")
    #output_match = os.path.join("analysis/", "bn_smallmodel_botherror_random_match.json")
    #output_nomatch = os.path.join("analysis/", "bn_smallmodel_botherror_random_nomatch.json")
    #output_partmatch = os.path.join("analysis/", "bn_smallmodel_botherror_random_partmatch.json")
    #compare_json(file_source, file_fix, output_match, output_nomatch, output_partmatch)

    #Golden结果，o1mini认为错误 & DeepSeek认为正确的，人随机抽取30，看一下对比情况
    #file_source = read_json(f"llm/correct_data_bn_gold_deepseek.json")
    #file_fix = read_json(f"annotated_final/annotations_single_gold_random_wlj_final.json")
    #output_match = os.path.join("analysis/", "bn_gold_deepseek_random_match.json")
    #output_nomatch = os.path.join("analysis/", "bn_gold_deepseek_random_nomatch.json")
    #output_partmatch = os.path.join("analysis/", "bn_gold_deepseek_random_partmatch.json")
    #compare_json(file_source, file_fix, output_match, output_nomatch, output_partmatch)

    #小模型产出结果，o1mini或DeepSeek认为正确的，人全部标注，看一下对比情况
    file_source1 = read_json(f"llm/correct_data_bn_smallmodel_deepseek.json")
    file_source2 = read_json(f"llm/correct_data_bn_smallmodel.json")
    file_source = file_source1 + file_source2
    file_fix = read_json(f"annotated_final/annotations_wlj_smallmodel_163_final.json")
    output_match = os.path.join("analysis/", "bn_smallmodel_dsright_match.json")
    output_nomatch = os.path.join("analysis/", "bn_smallmodel_dsright_nomatch.json")
    output_partmatch = os.path.join("analysis/", "bn_smallmodel_dsright_partmatch.json")
    compare_json(file_source, file_fix, output_match, output_nomatch, output_partmatch)
