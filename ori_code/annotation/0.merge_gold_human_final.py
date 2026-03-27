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

def transform_annodata_todict(data):
    data_dic = {}
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
    return data_dic

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
            #import pdb;pdb.set_trace()
    #print(f"org data argument num: {len(org_dic)}")
    return org_dic

def merge_org_into_dic(dic_data, org_data):
    """
    将org_data中的span补充进dic_data：
    - key不在dic_data：直接新增整条记录
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

# 比较输入文件correct_data和gold_data，对于不在correct_data中的gold_data，进行输出，输出格式需转成correct_data中的格式
def compare_gold_correct(gold_data, correct_data, domain):
    gold_keys = set(gold_data.keys())
    correct_keys = set(correct_data.keys())
    not_in_correct = gold_keys - correct_keys
    print(f"gold_data中不在correct_data中的记录有 {len(not_in_correct)} 条")
    out_list = []
    for key in not_in_correct:
        dict_item = {}
        dict_item["idx"] = gold_data[key]["idx"]
        dict_item["sen"] = gold_data[key]["sen"]
        dict_item["prd_word"] = gold_data[key]["prd_word"]
        dict_item["prd_lemma"] = gold_data[key]["prd_word"]
        dict_item["prd_sense"] = gold_data[key]["prd_sense"]
        dict_item["prd_idx"] = gold_data[key]["prd_idx"]
        dict_item["label"] = gold_data[key]["label"]
        dict_item["span"] = gold_data[key]["span"]
        dict_item["span_idx"] = gold_data[key]["span_idx"]
        dict_item["span_mean"] = gold_data[key]["span_mean"]
        out_list.append(dict_item)
    write_json(out_list, f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold_not_parse.json")

if __name__ == "__main__":
    domain = "tc"
    new_data = []
    dic_data = {}
    # step 1: 使用os.listdir 获得所有的人工标注的final数据
    file_gold_o1_wrong = f"anno/tc/annotation_tc_single_gold_201_wlj_final.json" # o1判为错的201条数据
    file_sm_o1_right = f"anno/tc/annotation_tc_smallmodel_removerepeate_322_wlj_final.json" # 小模型多产出的332条数据，不在golden中
    json_files = [file_gold_o1_wrong, file_sm_o1_right]
    for file_path in sorted(json_files):  # 排序以便有序处理
        print(f"  读取文件: {os.path.basename(file_path)}")
        data = read_json(file_path)
        print(f"数据个数为：{len(data['annotations'])}")
        new_data.extend(data['annotations'])
    print(f"\n总共读取了 {len(new_data)} 条数据")

    dic_data = transform_annodata_todict(new_data)

    print(f"\n去重后有 {len(dic_data)} 条数据")

    # step 2: 读取原始Golden数据，且o1-mini判断为正确的数据
    #o1_correct_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/llm_correct_data/correct_data_{domain}_gold.json") #bn
    o1_correct_data = read_json(f"llm/{domain}/correct_data_{domain}_gold.json")
    o1_correct_dict = transform_golddata_todict(o1_correct_data)
    print(f"o1-mini认为gold结果正确的数据有 {len(o1_correct_dict)} 条")

    # step 3: 合并o1mini认为正确的数据到dic_data
    dic_data = merge_org_into_dic(dic_data, o1_correct_dict)

    # step 4：读取原始gold数据
    org_data = transform_golddata_todict(read_json(f"final_data/{domain}/test_{domain}_4llm_core_gold.conll"))
    print(f"原始Golden数据条数为: {len(org_data)}")
    # step 5: 检查org_data中是否有在dic_data中没有的记录
    #org_not_in_dic = set(org_data.keys()) - set(dic_data.keys())
    #print(f"原始Golden在合并后数据中没有的记录有 {len(org_not_in_dic)} 条，具体如下:")
    # 打印这些记录
    #for key in org_not_in_dic:
    #    print(org_data[key])
    compare_gold_correct(org_data, dic_data, domain)
    
    # step6: 检查dic_data中是否有在org_data中没有的记录
    dic_not_in_org = set(dic_data.keys()) - set(org_data.keys())
    print(f"合并后数据在原始Golden中没有的记录有 {len(dic_not_in_org)} 条，具体如下:")
    # 打印这些记录
    for key in dic_not_in_org:
        print(dic_data[key])

    # 将dic_data的value字段写入到json文件中
    dic_data_values = list(dic_data.values())
    write_json(dic_data_values, f"new_test_set/final_annotated_all_{domain}.json") 
    print(f"合并后dic_data共 {len(dic_data)} 条")
