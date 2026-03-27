import os
import json
import pickle
from collections import defaultdict
from readline import read_history_file

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

# 转换golddata为dict，key为(sen, prd_word, prd_idx, label)，value为item_source，主要标注信息在span_idx字段
def transform_golddata_todict(org):
    org_dic = {}
    for i in range(len(org)):
        sen = org[i]['sen']
        prd_word = org[i]['prd_word']
        prd_idx = org[i]['prd_idx']
        label = org[i]['label']
        key = '\t'.join([sen, prd_word, str(prd_idx), label])
        if key not in org_dic.keys():
            org_dic[key] = org[i]
        else:
            print("org error! have repeat")
            import pdb;pdb.set_trace()
    #print(f"org data argument num: {len(org_dic)}")
    return org_dic
# 合并人工标注数据和o1判断为正确的gold数据
def merge_org_into_dic(dic_data, org_data):
    """
    将org_data中的span补充进dic_data：key不在dic_data：直接新增整条记录
    """
    added_new = 0
    added_span = 0

    for key, org_item in org_data.items():
        span = org_item.get('span_idx', None)
        if span is None:
            continue
        org_start, org_end = span[0], span[1]-1

        if key not in dic_data:
            # key不存在，新建整条记录
            sen, prd_word, prd_idx, label = key.split('\t')
            new_item = {
                'idx': org_item.get('idx', -1),
                'sentence': sen,
                'prd_word': prd_word,
                'prd_idx': int(prd_idx),
                'label': label,
                'span_mean': org_item.get('span_mean', ''),
                'type': 'gold_only',
                'timestamp': '',
                'grammar_status': '',
                'grammar_error_desc': '',
                'selected_spans': [{
                    'start': org_start,
                    'end': org_end,
                    'text': ' '.join(sen.split()[org_start:org_end+1]),
                    'models': ['gold_no_anno']
                }],
            }
            dic_data[key] = new_item
            added_new += 1

    print(f"从org_data新增了 {added_new} 条记录")
    return dic_data


def merge_selected_spans(existing_spans, new_spans):
    """合并两个selected_spans列表"""
    # 用(start, end, text)作为key建立索引
    span_dict = {}
    for span in existing_spans:
        key = (span['start'], span['end'], span['text'])
        span_dict[key] = span.copy()
    
    for new_span in new_spans:
        key = (new_span['start'], new_span['end'], new_span['text'])
        if key in span_dict:
            # 相同span，合并models（去重）
            existing_models = span_dict[key]['models']

            for model in new_span['models']:
                if model not in existing_models:
                    existing_models.append(model) # 原地修改
        else:
            # 不同span，直接添加
            span_dict[key] = new_span.copy()
    
    return list(span_dict.values())
# 转换annodata为dict，key为(sen, prd_word, prd_idx, label)，value为item_source，主要标注信息在selected_spans字段
def transform_annodata_todict(data):
    data_dic = {}
    merge_num = 0
    for item_source in data:
        sen = item_source["sentence"]
        prd_word = item_source["prd_word"]
        prd_idx = item_source["prd_idx"]
        label = item_source["label"]
        # span_idx = item_source["selected_spans"]
        #key_source = (sen, prd_word, prd_idx, label)
        key_source = '\t'.join([sen, prd_word, str(prd_idx), label])
        if key_source not in data_dic.keys():
            data_dic[key_source] = item_source 
        else:
            # print(data_dic[key_source]['selected_spans'])
            # print(item_source['selected_spans'])
            # 合并selected_spans
            org_span_num = len(data_dic[key_source]['selected_spans'])
            merged = merge_selected_spans(
                data_dic[key_source]['selected_spans'],
                item_source['selected_spans']
            )
            data_dic[key_source]['selected_spans'] = merged
            current_span_num = len(merged)
            if org_span_num != current_span_num:
                merge_num+=1
        
        # if data_dic[key_source].get('optional', None) !=None and data_dic[key_source].get('optional', None)==True:
    print(f"相同label，但不同批数据标注不同的instance个数：{merge_num},但目前已合并")   
    return data_dic




if __name__ == "__main__":
    domain = "bn"
    new_data = []
    dic_data = {}
    # 全部人工修正数据
    # json_files = ["annotated_final/annotations_gold_llm_not_parse_wlj_final.json", "annotated_final/annotations_single_gold_o1wrongdsright_93_wlj_final.json", "annotated_final/annotations_single_gold_random_wlj_final.json", "annotated_final/annotations_single_gold_wlj_final.json", "annotated_final/annotations_wlj_smallmodel_163_final.json", "annotated_final/annotation_smallmodel_overlap_wlj_final.json"]
    json_files = ["annotated_final/annotations_gold_llm_not_parse_wlj_final.json", 
    "annotated_final/annotations_single_gold_o1wrongdsright_93_wlj_final.json", 
    "annotated_final/annotations_single_gold_random_wlj_final.json", 
    "annotated_final/annotations_single_gold_wlj_final.json", 
    "annotated_final/annotations_wlj_smallmodel_163_final.json", 
    "annotated_final/annotation_smallmodel_overlap_wlj_final.json", 
    "annotated_final/annotation_smnotrecall_21_wlj_final.json",
    "annotated_final/annotation_smnotrecallright_110_wlj_final.json"]
    for file_path in sorted(json_files):  # 排序以便有序处理
        print(f"  读取文件: {os.path.basename(file_path)}")
        data = read_json(file_path)
        print(f"数据个数为：{len(data['annotations'])}")
        new_data.extend(data['annotations'])
    print(f"\n总共读取了 {len(new_data)} 条数据")

    anno_data = transform_annodata_todict(new_data)
    print(f"\n去重后有 {len(anno_data)} 条数据")
    num_none = 0
    for key in anno_data:
        if len(anno_data[key]['selected_spans'])==0:
            num_none+=1
    print(f"无论元的句子为：{num_none}")


    # step 2: 读取原始Golden数据，且o1-mini判断为正确的数据
    o1_correct_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/llm_correct_data/correct_data_{domain}_gold.json")
    o1_correct_dict = transform_golddata_todict(o1_correct_data)
    print(f"o1-mini认为gold结果正确的数据有 {len(o1_correct_dict)} 条")

    # step 3: 合并o1mini认为正确的数据到dic_data
    all_data = merge_org_into_dic(anno_data, o1_correct_dict)
    print(f'final_data: {len(all_data)}')
    write_json(all_data, f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_500_core_final_v4.json")

    # write_json(anno_data, f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_500_core_final_V3.json")

    # # step 2: 读取原始Golden数据，且o1-mini判断为正确的数据
    # o1_correct_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/llm_correct_data/correct_data_{domain}_gold.json")
    # o1_correct_dict = transform_golddata_todict(o1_correct_data)
    # print(f"o1-mini认为gold结果正确的数据有 {len(o1_correct_dict)} 条")

    # # step 3: 合并o1mini认为正确的数据到dic_data
    # all_data = merge_org_into_dic(anno_data, o1_correct_dict)
    # write_json(all_data, f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_500_core_final.json")

    # # step 4：读取原始gold数据
    # org_data = transform_golddata_todict(read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/test_{domain}_4llm_core_gold.conll"))
    # print(f"原始Golden数据条数为: {len(org_data)}")
    
    # # step 5:
    # out_file = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_500_core_final_errorandmiss.json"
    # compute_gold_errorrate(org_data, all_data, out_file)
    

    # # step6:对于可标可不标注的结果进行merge（注意tc的时候应该就不需要这一步了） 这里需要看下是否需要
    # pre_core_final =  read_json(f"analysis/test_{domain}_500_core_final.json")
    # print(f"pre final data num : {len(pre_core_final)}")
    # sm_norecallinfinal = read_json(f'annotated_final/annotation_smnotrecall_21_wlj_final.json')
    # sm_dic = transform_reviweddata_todict(sm_norecallinfinal)
    # merge_other_into_final(pre_core_final, sm_dic)

