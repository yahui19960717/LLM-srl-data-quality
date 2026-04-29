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

# 合并两个selected_spans列表，相同span合并models（去重），不同span直接添加
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
def transform_annodata_todict(data, opt_outfile):
    data_dic = {}
    opt_dict = {}
    merge_num = 0
    for item_source in data:
        sen = item_source["sentence"]
        prd_word = item_source["prd_word"]
        prd_idx = item_source["prd_idx"]
        label = item_source["label"]
        optional_flag = item_source.get('optional', False)
        # span_idx = item_source["selected_spans"]
        #key_source = (sen, prd_word, prd_idx, label)
        key_source = '\t'.join([sen, prd_word, str(prd_idx), label])
        if key_source not in data_dic.keys():
            data_dic[key_source] = item_source
            if optional_flag:
                opt_dict[key_source] = item_source
        else:
            org_span_num = len(data_dic[key_source]['selected_spans'])
            merged = merge_selected_spans(
                data_dic[key_source]['selected_spans'],
                item_source['selected_spans']
            )
            data_dic[key_source]['selected_spans'] = merged
            if optional_flag:
                opt_dict[key_source] = item_source
            current_span_num = len(merged)
            if org_span_num != current_span_num:
                merge_num+=1     
    write_json(opt_dict, opt_outfile)
    print(f"相同label，但不同批数据标注不同的instance个数：{merge_num},但目前已合并") 
    return data_dic

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
            #import pdb;pdb.set_trace()
    #print(f"org data argument num: {len(org_dic)}")
    return org_dic

#构建annodata的item，包含idx、sen、prd_word、prd_idx、label、span_mean、gold、selected_span，主要是原始标注错误和漏标注的，用于分析case用
def construct_item(data, gold_span):
    dict_item = {}
    dict_item["idx"] = data.get('idx', -1)
    dict_item["sen"] = data.get('sentence', '')
    dict_item["prd_word"] = data.get('prd_word', '')
    dict_item["prd_idx"] = data.get('prd_idx', -1)
    dict_item["label"] = data.get('label', '')
    dict_item["span_mean"] = data.get('span_mean', '')
    dict_item["gold"] = gold_span
    dict_item["selected_span"] = data.get('selected_spans', None)
    return dict_item

# 写一个函数，名称为compute_gold_errorrate, 输入为gold_data_dict、annodata_dict以及输出文件, 最终将人工修正数据写入输出文件
# 设置变量total_num等于gold_data_dict的大小，error_num表示gold数据与人工不匹配的，multi_candidate_num表示人工标注了多个候选，miss_num表示人工标注多出来的数据
# 先遍历annodata_dict，如果不在gold_data_dict中，miss_num+=1；如果在gold_data_dict中，判断selected_spans与span_idx是否匹配，若不匹配，则error_num+=1，若selected_spans有多个结果，则multi_candidate_num+=1
def compute_gold_errorrate(gold_data_dict, annodata_dict, output_file_miss, output_file_error, output_file_multi):
    total_num = len(gold_data_dict)
    error_num = 0
    multi_candidate_num = 0
    miss_num = 0
    match_num = 0
    output_list_miss = []
    output_list_error = []
    output_list_multi = []
    for key, ann_item in annodata_dict.items():
        if key not in gold_data_dict.keys(): # 人工标注数据不在gold中，属于gold漏标注
            if ann_item.get('selected_spans', None) is not None:
                dict_item = construct_item(ann_item, None)
                output_list_miss.append(dict_item)
                miss_num += 1
            #print(f"miss: {key}")
            continue
        gold_item = gold_data_dict[key]
        gold_span = gold_item.get('span_idx', None)
        if gold_span is None:
            if ann_item.get('selected_spans', None) is None:
                #print("both none")
                continue
            dict_item = construct_item(ann_item, None)
            output_list_miss.append(dict_item)
            miss_num += 1
            continue
        # 对比gold span与人工标注的selected_spans是否匹配
        gold_start, gold_end = gold_span[0], gold_span[1]-1
        ann_spans = ann_item.get('selected_spans', None)
        if ann_spans is None: #人工没有标注，属于gold错误标注
            error_num += 1
            dict_item = construct_item(ann_item, gold_span)
            output_list_error.append(dict_item)
            continue
        if len(ann_spans) > 1:
            multi_candidate_num += 1
            dict_item = construct_item(ann_item, gold_span)
            output_list_multi.append(dict_item)
        # 遍历selected_spans，依次取start和end与gold_start和gold_end进行对比，如果完全没有匹配上，则error_num+=1
        match_flag = 0
        for ann_span in ann_spans:
            ann_start, ann_end = ann_span['start'], ann_span['end']
            if ann_start == gold_start and ann_end == gold_end:
                match_num += 1
                match_flag = 1
                break
        if match_flag == 0:
            error_num += 1
            dict_item = construct_item(ann_item, gold_span)
            output_list_error.append(dict_item)
    # 打印error_num、multi_candidate_num、miss_num、total_num
    print(f"gold数据中总共有 {total_num} 条记录，其中{match_num}条匹配上，标注错误的有 {error_num} 条，人工标注了多个候选的有 {multi_candidate_num} 条，漏标注的有 {miss_num} 条")
    write_json(output_list_error, output_file_error)
    write_json(output_list_multi, output_file_multi)
    write_json(output_list_miss, output_file_miss)

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
                'optional': False,
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

# 处理bn数据，统计不同标注数据的准确率、漏标率等
def process_bn_data(domain):
    new_data = []
    dic_data = {}
    # step 1: 使用os.listdir 获得所有的人工标注的final数据
    # 来自gold的人工修正数据
    file_gold_o1_right = f"llm/correct_data_{domain}_gold.json" #o1判别为对的数据，1893个角色，人工抽查准确率为93.3%
    file_gold_o1_wrong_ds_wrong = f"annotated_final/annotations_single_gold_wlj_final.json" # o1和ds都判为错的113条数据
    file_gold_not_parser = f"annotated_final/annotations_gold_llm_not_parse_wlj_final.json" # LLM结果没有解析正确的3条数据
    file_gold_o1_wrong_ds_right1 = f"annotated_final/annotations_single_gold_random_wlj_final.json" #o1判为错、ds判为正确的数据，第一批随机30
    file_gold_o1_wrong_ds_right2 = f"annotated_final/annotations_single_gold_o1wrongdsright_93_wlj_final.json" #1判为错、ds判为正确的数据，第二批数据，剩余93条
    file_gold_sm_nocall1 = f"annotated_final/annotation_smnotrecall_21_wlj_final.json" # 小模型未召回、大模型认为错误的数据，21条
    file_gold_sm_nocall2 = f"annotated_final/annotation_smnotrecallright_random30_wlj_final.json" #小模型未召回、大模型认为对的数据，随机30条
    file_gold_sm_nocall3 = f"annotated_final/annotation_smnotrecallright_110_wlj_final.json" #小模型未召回、大模型认为对的数据，剩余110条
    json_gold_files = [file_gold_o1_wrong_ds_wrong, file_gold_not_parser, file_gold_o1_wrong_ds_right1, file_gold_o1_wrong_ds_right2, file_gold_sm_nocall1, file_gold_sm_nocall2, file_gold_sm_nocall3]
    #json_gold_files = [file_gold_o1_wrong_ds_wrong, file_gold_not_parser, file_gold_o1_wrong_ds_right1, file_gold_o1_wrong_ds_right2]
    # 来自小模型多产出的数据
    file_sm_no_overlap = f"annotated_final/annotations_wlj_smallmodel_163_final.json" # 小模型多产出的163条数据，不在golden中，且仅属于semicrf或treecrf
    file_sm_overlap = f"annotated_final/annotation_smallmodel_overlap_wlj_final.json" # 小模型多产出的179条数据，不在golden中，semicrf和treecrf相同的部分
    json_sm_files = [file_sm_no_overlap, file_sm_overlap]
    # 全部人工修正数据
    json_files = json_gold_files + json_sm_files
    #json_files = json_gold_files
    for file_path in sorted(json_files):  # 排序以便有序处理
        print(f"  读取文件: {os.path.basename(file_path)}")
        data = read_json(file_path)
        print(f"数据个数为：{len(data['annotations'])}")
        new_data.extend(data['annotations'])
    print(f"\n总共读取了 {len(new_data)} 条数据")

    opt_outfile = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_optional.json"
    anno_data = transform_annodata_todict(new_data, opt_outfile)
    print(f"\n去重后有 {len(anno_data)} 条数据")

    # step 2：读取原始gold数据
    org_data = transform_golddata_todict(read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold.conll"))
    print(f"原始Golden数据条数为: {len(org_data)}")
    
    # step 3:
    out_file_error = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_4llm_core_gold_error.json"
    out_file_multi = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_4llm_core_gold_multi.json"
    out_file_miss = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_4llm_core_gold_miss.json"
    compute_gold_errorrate(org_data, anno_data, out_file_miss, out_file_error, out_file_multi)

# 合并bn数据，合并人工标注数据和o1判断为正确的gold数据，得到最终的新的集合
def merge_bn_data(domain):
    new_data = []
    dic_data = {}
    # step 1: 使用os.listdir 获得所有的人工标注的final数据
    # 来自gold的人工修正数据
    file_gold_o1_right = f"llm/correct_data_{domain}_gold.json" #o1判别为对的数据，1893个角色，人工抽查准确率为93.3%
    file_gold_o1_wrong_ds_wrong = f"annotated_final/annotations_single_gold_wlj_final.json" # o1和ds都判为错的113条数据
    file_gold_not_parser = f"annotated_final/annotations_gold_llm_not_parse_wlj_final.json" # LLM结果没有解析正确的3条数据
    file_gold_o1_wrong_ds_right1 = f"annotated_final/annotations_single_gold_random_wlj_final.json" #o1判为错、ds判为正确的数据，第一批随机30
    file_gold_o1_wrong_ds_right2 = f"annotated_final/annotations_single_gold_o1wrongdsright_93_wlj_final.json" #1判为错、ds判为正确的数据，第二批数据，剩余93条
    file_gold_sm_nocall1 = f"annotated_final/annotation_smnotrecall_21_wlj_final.json" # 小模型未召回、大模型认为错误的数据，21条
    file_gold_sm_nocall2 = f"annotated_final/annotation_smnotrecallright_random30_wlj_final.json" #小模型未召回、大模型认为对的数据，随机30条
    file_gold_sm_nocall3 = f"annotated_final/annotation_smnotrecallright_110_wlj_final.json" #小模型未召回、大模型认为对的数据，剩余110条
    json_gold_files = [file_gold_o1_wrong_ds_wrong, file_gold_not_parser, file_gold_o1_wrong_ds_right1, file_gold_o1_wrong_ds_right2, file_gold_sm_nocall1, file_gold_sm_nocall2, file_gold_sm_nocall3]
    # 来自小模型多产出的数据
    file_sm_no_overlap = f"annotated_final/annotations_wlj_smallmodel_163_final.json" # 小模型多产出的163条数据，不在golden中，且仅属于semicrf或treecrf
    file_sm_overlap = f"annotated_final/annotation_smallmodel_overlap_wlj_final.json" # 小模型多产出的179条数据，不在golden中，semicrf和treecrf相同的部分
    json_sm_files = [file_sm_no_overlap, file_sm_overlap]
    # 全部人工修正数据
    json_files = json_gold_files + json_sm_files
    for file_path in sorted(json_files):  # 排序以便有序处理
        print(f"  读取文件: {os.path.basename(file_path)}")
        data = read_json(file_path)
        print(f"数据个数为：{len(data['annotations'])}")
        new_data.extend(data['annotations'])
    print(f"\n总共读取了 {len(new_data)} 条数据")

    opt_outfile = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_optional.json"
    anno_data = transform_annodata_todict(new_data, opt_outfile)
    print(f"\n去重后有 {len(anno_data)} 条数据")

    # step 2: 读取原始Golden数据，且o1-mini判断为正确的数据
    o1_correct_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/llm_correct_data/correct_data_{domain}_gold.json")
    o1_correct_dict = transform_golddata_todict(o1_correct_data)
    print(f"o1-mini认为gold结果正确的数据有 {len(o1_correct_dict)} 条")

    # step 3: 合并o1mini认为正确的数据到dic_data
    all_data = merge_org_into_dic(anno_data, o1_correct_dict)
    print(f"合并后有 {len(all_data)} 条数据")
    write_json(all_data, f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/new_test_set/test_{domain}_500_core_final.json")
    
    # step 4：读取原始gold数据
    org_data = transform_golddata_todict(read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold.conll"))
    print(f"原始Golden数据条数为: {len(org_data)}")
    
    # step 5: 计算最终数据的错误率和漏标率，以及保存错误和漏标注的数据，用于后续分析
    out_file_miss = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_miss.json"
    out_file_error = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_error.json"
    out_file_multi = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_multi.json"
    compute_gold_errorrate(org_data, all_data, out_file_miss, out_file_error, out_file_multi)

# 处理tc数据，统计不同标注数据的准确率、漏标率等
def process_tc_data(domain):
    new_data = []
    dic_data = {}
    # step 1: 使用os.listdir 获得所有的人工标注的final数据
    # 来自gold的人工修正数据
    file_gold_o1_right = f"llm/tc/correct_data_{domain}_gold.json" #o1判别为对的数据，
    file_gold_o1_wrong = f"anno/tc/annotation_tc_single_gold_201_wlj_final.json" # o1判为错的201条数据
    file_gold_not_parser = f"anno/tc/annotation_tc_gold_llm_not_parse_wlj_final.json" # LLM结果没有解析正确的3条数据
    
    json_gold_files = [file_gold_o1_wrong, file_gold_not_parser]
    # 来自小模型多产出的数据
    file_sm_o1_right = f"anno/tc/annotation_tc_smallmodel_removerepeate_322_wlj_final.json" # 小模型多产出的332条数据，不在golden中
    json_sm_files = [file_sm_o1_right]
    # 全部人工修正数据
    json_files = json_gold_files + json_sm_files
    #json_files = json_sm_files
    for file_path in sorted(json_files):  # 排序以便有序处理
        print(f"  读取文件: {os.path.basename(file_path)}")
        data = read_json(file_path)
        print(f"数据个数为：{len(data['annotations'])}")
        new_data.extend(data['annotations'])
    print(f"\n总共读取了 {len(new_data)} 条数据")

    opt_outfile = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_optional.json"
    anno_data = transform_annodata_todict(new_data, opt_outfile)
    print(f"\n去重后有 {len(anno_data)} 条数据")

    # step 2：读取原始gold数据
    org_data = transform_golddata_todict(read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold.conll"))
    print(f"原始Golden数据条数为: {len(org_data)}")
    
    # step 3:
    out_file_miss = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_4llm_core_gold_miss.json"
    out_file_error = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_4llm_core_gold_error.json"
    out_file_multi = f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_4llm_core_gold_multi.json"
    compute_gold_errorrate(org_data, anno_data, out_file_miss, out_file_error, out_file_multi)

# 合并tc数据，合并人工标注数据和o1判断为正确的gold数据，得到最终的新的集合
def merge_tc_data(domain):
    new_data = []
    dic_data = {}
    # step 1: 使用os.listdir 获得所有的人工标注的final数据
    # 来自gold的人工修正数据
    file_gold_o1_right = f"llm/tc/correct_data_{domain}_gold.json" #o1判别为对的数据，
    file_gold_o1_wrong = f"anno/tc/annotation_tc_single_gold_201_wlj_final.json" # o1判为错的201条数据
    file_gold_not_parser = f"anno/tc/annotation_tc_gold_llm_not_parse_wlj_final.json" # LLM结果没有解析正确的3条数据
    # 来自小模型多产出的数据
    file_sm_o1_right = f"anno/tc/annotation_tc_smallmodel_removerepeate_322_wlj_final.json" # 小模型多产出的332条数据，不在golden中
    # 全部人工修正数据
    json_files = [file_gold_o1_wrong, file_gold_not_parser, file_sm_o1_right]
    
    for file_path in sorted(json_files):  # 排序以便有序处理
        print(f"  读取文件: {os.path.basename(file_path)}")
        data = read_json(file_path)
        print(f"数据个数为：{len(data['annotations'])}")
        new_data.extend(data['annotations'])
    print(f"\n总共读取了 {len(new_data)} 条数据")

    opt_outfile = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_optional.json"
    anno_data = transform_annodata_todict(new_data, opt_outfile)
    print(f"\n去重后有 {len(anno_data)} 条数据")

    # step 2: 读取原始Golden数据，且o1-mini判断为正确的数据
    o1_correct_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/llm_correct_data/correct_data_{domain}_gold.json")
    o1_correct_dict = transform_golddata_todict(o1_correct_data)
    print(f"o1-mini认为gold结果正确的数据有 {len(o1_correct_dict)} 条")

    # step 3: 合并o1mini认为正确的数据到dic_data
    all_data = merge_org_into_dic(anno_data, o1_correct_dict)
    print(f"合并后有 {len(all_data)} 条数据")
    write_json(all_data, f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/new_test_set/test_{domain}_500_core_final.json")
    
    # step 4：读取原始gold数据
    org_data = transform_golddata_todict(read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold.conll"))
    print(f"原始Golden数据条数为: {len(org_data)}")
    
    # step 5: 计算最终数据的错误率和漏标率，以及保存错误和漏标注的数据，用于后续分析
    out_file_miss = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_miss.json"
    out_file_error = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_error.json"
    out_file_multi = f"/data/ljwang/span-SRL-LLM/ori_code/prompt_srl/analysis_data/test_{domain}_multi.json"
    compute_gold_errorrate(org_data, all_data, out_file_miss, out_file_error, out_file_multi)


if __name__ == "__main__":
    domain = "tc"
    type = "merge"
    if domain == "bn":
        if type == "merge":
            merge_bn_data(domain)
        else:
            process_bn_data(domain)
    elif domain == "tc":
        if type == "merge":
            merge_tc_data(domain)
        else:
            process_tc_data(domain)
