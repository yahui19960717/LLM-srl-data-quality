"""
那我们可以先使用这500测试集：原来的标注及现在修正后的标注，重新评估一下两个小模型的结果，只看核心角色，看一下精确率、召回率和F1的变化

1. 目前我需要把所有的标注数据都放在一起作为最终的标注结果(current gold)（500句）

2. 获得原来的semicrf预测和treecrf预测(注意，这里的semicrf和treecrf都是带be动词的)

3. 写评估脚本：目前预测的PRF值 (current gold) :注意，这里的评估需要更改，只要有一个符合的就算对

4. 写评估脚本：原来预测的PRF值（previous gold)


"""
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

def write_pickle(sentences, path):
    with open(path, 'wb') as f:
        pickle.dump(sentences, f)

def get_dic(data):
    data_dic = {}
    for item_source in data:
        sen = item_source["sentence"]
        prd_word = item_source["prd_word"]
        prd_idx = item_source["prd_idx"]
        label = item_source["label"]
        # span_idx = item_source["selected_spans"]
        key_source = (sen, prd_word, prd_idx, label)
        if key_source not in data_dic.keys():
            data_dic[key_source] = item_source     
    return data_dic

def get_dic_ignore_unimportant(data):
    data_dic = {}
    grammar_error_num, no_arg, multi_arg, repeat = 0, 0, 0, 0
    grammar_error_instance, noarg_instance= [], []
    for item_source in data:
        sen = item_source["sentence"]
        prd_word = item_source["prd_word"]
        prd_idx = item_source["prd_idx"]
        label = item_source["label"]
        # span_idx = item_source["selected_spans"]
        key_source = (sen, prd_word, prd_idx, label)
        if item_source['grammar_status'] == "有语法错误":
            grammar_error_num+= 1
            grammar_error_instance.append(item_source)
        else:
            if len(item_source["selected_spans"]) == 0: # 无论元
                no_arg += 1
                noarg_instance.append(item_source)
            else:
                if key_source not in data_dic.keys():
                    data_dic[key_source] = item_source 
                    if len(item_source["selected_spans"])>=2:
                        multi_arg += 1
                else:
                    repeat += 1
                    # print("error!")

    print("*"*50)
    print(f'统计Corrected data: 修正的总个数为{len(data)}')
    print(f'语法错误的数据个数：{grammar_error_num}')
    print(f'无论元的数据个数: {no_arg}')
    print(f'多候选的数据个数: {multi_arg}')
    print(f'去重的个数为: {repeat}')
    print(f"[去重后有 {len(data_dic)} 条数据]")
    print("*"*50)
    return data_dic, grammar_error_num, no_arg, multi_arg

def get_all_orggold(org):
    org_dic = {}
    for i in range(len(org)):
        sen = org[i]['sen']
        prd_word = org[i]['prd_word']
        prd_idx = org[i]['prd_idx']
        label = org[i]['label']
        key = (sen, prd_word, prd_idx, label)
        if key not in org_dic.keys():
            org_dic[key] = org[i]
        else:
            print("org error! have repeat")
            import pdb;pdb.set_trace()
    return org_dic


def merge_org_into_dic(dic_data, org_data):
    """
    将o1mini 认为org_data中正确的span补充进dic_data：
    - key不在dic_data：直接新增整条记录
    """
    added_new = 0
    added_span = 0
    repeat = 0
    for key, org_item in org_data.items():
        span = org_item.get('span_idx', None) # 
        assert span != None
        assert span != []
        org_start, org_end = span[0], span[1]-1

        if key not in dic_data: # 只是把大模型认为正确的gold加上
            # key不存在，新建整条记录
            sen, prd_word, prd_idx, label = key
            new_item = {
                'idx': org_item.get('idx', -1),
                'sentence': sen,
                'prd_word': prd_word,
                'prd_idx': prd_idx,
                'label': label,
                'span_mean': org_item.get('span_mean', ''),
                'type': 'gold_only',
                'timestamp': '',
                'grammar_status': '',
                'grammar_error_desc': '',
                'selected_spans': [{
                    'start': org_start,
                    'end': org_end,
                    'text': ' '.join(sen.split()[org_start:org_end]),
                    'models': ['gold_no_anno']
                }],
            }
            dic_data[key] = new_item
            added_new += 1
        else:
            repeat += 1 # 主要是有随机查找的o1right的30个
            # # key已存在，检查该span是否已在selected_spans中
            import pdb;pdb.set_trace()
            # existing_spans = dic_data[key].get('selected_spans', [])
            # already_exists = any(
            #     s['start'] == org_start and s['end'] == org_end
            #     for s in existing_spans
            # )
            # if not already_exists:
            #     sen = key[0]
            #     try:
            #         existing_spans.append({
            #             'start': org_start,
            #             'end': org_end,
            #             'text': ' '.join(sen.split()[org_start:org_end]),
            #             'models': ['gold']
            #         })
            #     except:
            #         import pdb;pdb.set_trace()
            #     dic_data[key]['selected_spans'] = existing_spans
            #     added_span += 1

    # print(f"从org_data新增了 {added_new} 条记录，向已有记录追加了 {added_span} 个span")
    print(f"从org_data新增了 {added_new} 条记录")
    print(f'原始标注中的论元曾在小模型中出现过的个数为：{sm_have}')
    import pdb;pdb.set_trace()
    return dic_data

def build_gold_spans_from_dic(dic_data):
    """
    从dic_data构建current gold标准：
    key = (sen, prd_word, prd_idx, label)
    value = selected_spans列表（人工审核后的正确span边界集合）
    
    返回结构：
    {
      (sen, prd_word, prd_idx): {
          label: [(start, end), ...]
      }
    }
    """
    gold_dict = defaultdict(lambda: defaultdict(list))
    for (sen, prd_word, prd_idx, label), item in dic_data.items():
        for span in item.get('selected_spans', []):
            gold_dict[(sen, prd_word, prd_idx)][label].append((span['start'], span['end']))
    return gold_dict

def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

def spans_to_set(label_spans_dict):
    """
    将 {label: [(start,end),...]} 转换为 {(label, start, end)} 集合，仅保留ARG0-ARG5
    """
    target_labels = {f'ARG{i}' for i in range(6)}
    result = set()
    for label, spans in label_spans_dict.items():
        if label not in target_labels:
            continue
        for span in spans:
            result.add((label, span[0], span[1]))
    return result

def evaluate_model(model_name, pred_field, eval_data, gold_dict):
    """
    根据dic_data构建的gold_dict评估某个小模型的ARG0-ARG5 PRF。
    
    eval_data: read_json读取的列表，每个元素包含 sen/prd_word/pred.idx/gold/semicrf_label/treecrf_label
    gold_dict: build_gold_spans_from_dic的输出
    pred_field: 'semicrf_label' 或 'treecrf_label'
    """
    target_labels = {f'ARG{i}' for i in range(6)}
    
    # 整体计数
    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    # 每个标签计数
    per_label = {lbl: {'tp': 0, 'fp': 0, 'fn': 0} for lbl in target_labels}

    matched = 0
    unmatched_pred = 0

    for item in eval_data: # 谓词级别的处理
        sen      = item['sen']
        prd_word = item['prd_word']
        prd_idx  = item.get('pred.idx', item.get('prd_idx', -1))
        
        lookup_key = (sen, prd_word, prd_idx)
        
        if lookup_key not in gold_dict:
            # 该谓词未在dic_data中，跳过 # 包含了之前处理的be动词、只有be动词的谓词等等
            unmatched_pred += 1
            # print(lookup_key)
            continue
        matched += 1

        gold_spans = gold_dict[lookup_key]   # {label: [(start,end),...]}
        pred_spans = item.get(pred_field, {}) # {label: [start, end]}  注意：单个span是列表[start,end]

        gold_set = spans_to_set(gold_spans)
        
        # 将pred转成 {label: (start, end)}，只处理ARG0-ARG5
        pred_dict = {}
        for label, span in pred_spans.items(): 
            if label not in target_labels:
                continue
            if isinstance(span[0], list):
                # 兼容多span格式
                pred_dict[label] = [(s[0], s[1]-1) for s in span] # 原始的处理结果中需要 end-1
            else:
                pred_dict[label] = [(span[0], span[1]-1)]  # 原始的处理结果中需要 end-1

        # 以label为单位计算TP/FP/FN
        # gold中出现的label集合 与 pred中出现的label集合
        gold_labels = {lbl for lbl in gold_spans if lbl in target_labels}
        pred_labels = set(pred_dict.keys())
        all_labels  = gold_labels | pred_labels # 所有可能的labels

        for lbl in all_labels:
            gold_set = set(gold_spans.get(lbl, []))  # 该label的所有合法span
            pred_list = pred_dict.get(lbl, [])        # 该label的预测span（最多一个）
            # import pdb;pdb.set_trace()
            if pred_list:
                assert len(pred_list) == 1
                pred_span = pred_list[0]  # 预测中该label只有一个span
                if pred_span in gold_set:
                    # 命中gold中任意一个span → TP
                    overall['tp'] += 1
                    per_label[lbl]['tp'] += 1
                else:
                    # 未命中 → FP + FN（预测了但预测错了，gold该label也没被覆盖）
                    overall['fp'] += 1
                    per_label[lbl]['fp'] += 1
                    if lbl in gold_labels:
                        overall['fn'] += 1
                        per_label[lbl]['fn'] += 1
            else:
                # 该label有gold但没有预测 → FN
                if lbl in gold_labels:
                    overall['fn'] += 1
                    per_label[lbl]['fn'] += 1

    print(f"\n{'='*75}")
    print(f"模型: {model_name}  |  匹配谓词: {matched}  |  未命中: {unmatched_pred}")
    print(f"{'='*75}")
    print(f"{'标签':<12} {'Precision(%)':>8} {'Recall(%)':>8} {'F1(%)':>6}  {'TP':>5}   {'FP':>5}  {'FN':>5}")
    print(f"{'-'*75}")

    p, r, f = compute_prf(overall['tp'], overall['fp'], overall['fn'])
    print(f"{'Overall':<11} {p*100:>10.2f} {r*100:>10.2f} {f*100:>10.2f}"
          f"  {overall['tp']:>6}  {overall['fp']:>5}  {overall['fn']:>5}")
    print(f"{'-'*75}")

    for lbl in sorted(target_labels):
        c = per_label[lbl]
        if c['tp'] + c['fp'] + c['fn'] == 0:
            continue
        p, r, f = compute_prf(c['tp'], c['fp'], c['fn'])
        print(f"{lbl:<11} {p*100:>10.2f} {r*100:>10.2f} {f*100:>10.2f}"
              f"  {c['tp']:>6}  {c['fp']:>5}  {c['fn']:>5}")
        # print(f"{lbl:<11} {p:>10.5f} {r:>10.5f} {f:>10.5f}"
        #       f"  {c['tp']}  {c['fp']}  {c['fn']}")
    print(f"{'='*75}")

def evaluate_model_gold(model_name, pred_field, eval_data, org_dict, gold_dict):
    """
    根据dic_data构建的gold_dict评估某个小模型的ARG0-ARG5 PRF。
    
    eval_data: read_json读取的列表，每个元素包含 sen/prd_word/pred.idx/gold/semicrf_label/treecrf_label
    gold_dict: build_gold_spans_from_dic的输出
    pred_field: 'semicrf_label' 或 'treecrf_label'
    """
    target_labels = {f'ARG{i}' for i in range(6)}
    
    # 整体计数
    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    # 每个标签计数
    per_label = {lbl: {'tp': 0, 'fp': 0, 'fn': 0} for lbl in target_labels}

    matched = 0
    unmatched_pred = 0

    for item in eval_data: # 谓词级别的处理
        sen      = item['sen']
        prd_word = item['prd_word']
        prd_idx  = item.get('pred.idx', item.get('prd_idx', -1))
        
        lookup_key = (sen, prd_word, prd_idx)
        
        if lookup_key not in gold_dict:
            # 该谓词未在dic_data中，跳过 # 包含了之前处理的be动词、只有be动词的谓词等等
            unmatched_pred += 1
            # print(lookup_key)
            continue
        matched += 1

        gold_spans = org_dict[lookup_key]   # {label: [(start,end),...]}
        pred_spans = item.get(pred_field, {}) # {label: [start, end]}  注意：单个span是列表[start,end]

        gold_set = spans_to_set(gold_spans)
        
        # 将pred转成 {label: (start, end)}，只处理ARG0-ARG5
        pred_dict = {}
        for label, span in pred_spans.items(): 
            if label not in target_labels:
                continue
            if isinstance(span[0], list):
                # 兼容多span格式
                pred_dict[label] = [(s[0], s[1]-1) for s in span] # 原始的处理结果中需要 end-1
            else:
                pred_dict[label] = [(span[0], span[1]-1)]  # 原始的处理结果中需要 end-1

        # 以label为单位计算TP/FP/FN
        # gold中出现的label集合 与 pred中出现的label集合
        gold_labels = {lbl for lbl in gold_spans if lbl in target_labels}
        pred_labels = set(pred_dict.keys())
        all_labels  = gold_labels | pred_labels # 所有可能的labels

        for lbl in all_labels:
            gold_set = set(gold_spans.get(lbl, []))  # 该label的所有合法span
            pred_list = pred_dict.get(lbl, [])        # 该label的预测span（最多一个）
            # import pdb;pdb.set_trace()
            if pred_list:
                assert len(pred_list) == 1
                pred_span = pred_list[0]  # 预测中该label只有一个span
                if pred_span in gold_set:
                    # 命中gold中任意一个span → TP
                    overall['tp'] += 1
                    per_label[lbl]['tp'] += 1
                else:
                    # 未命中 → FP + FN（预测了但预测错了，gold该label也没被覆盖）
                    overall['fp'] += 1
                    per_label[lbl]['fp'] += 1
                    if lbl in gold_labels:
                        overall['fn'] += 1
                        per_label[lbl]['fn'] += 1
            else:
                # 该label有gold但没有预测 → FN
                if lbl in gold_labels:
                    overall['fn'] += 1
                    per_label[lbl]['fn'] += 1

    print(f"\n{'='*75}")
    print(f"模型: {model_name}  |  匹配谓词: {matched}  |  未命中: {unmatched_pred}")
    print(f"{'='*75}")
    print(f"{'标签':<12} {'Precision(%)':>8} {'Recall(%)':>8} {'F1(%)':>6}  {'TP':>5}   {'FP':>5}  {'FN':>5}")
    print(f"{'-'*75}")

    p, r, f = compute_prf(overall['tp'], overall['fp'], overall['fn'])
    print(f"{'Overall':<11} {p*100:>10.2f} {r*100:>10.2f} {f*100:>10.2f}"
          f"  {overall['tp']:>6}  {overall['fp']:>5}  {overall['fn']:>5}")
    print(f"{'-'*75}")

    for lbl in sorted(target_labels):
        c = per_label[lbl]
        if c['tp'] + c['fp'] + c['fn'] == 0:
            continue
        p, r, f = compute_prf(c['tp'], c['fp'], c['fn'])
        print(f"{lbl:<11} {p*100:>10.2f} {r*100:>10.2f} {f*100:>10.2f}"
              f"  {c['tp']:>6}  {c['fp']:>5}  {c['fn']:>5}")
        # print(f"{lbl:<11} {p:>10.5f} {r:>10.5f} {f:>10.5f}"
        #       f"  {c['tp']}  {c['fp']}  {c['fn']}")
    print(f"{'='*75}")

def build_gold_dict_from_eval(eval_data):
    """
    直接从eval_data中的gold字段构建gold_dict
    格式: {(sen, prd_word, prd_idx): {label: [(start, end)]}}
    """
    gold_dict = defaultdict(lambda: defaultdict(list))
    for item in eval_data:
        sen     = item['sen']
        prd_word = item['prd_word']
        prd_idx  = item.get('pred.idx', item.get('prd_idx', -1))
        key = (sen, prd_word, prd_idx)
        for label, span in item.get('gold', {}).items():
            if isinstance(span[0], list):
                for s in span:
                    gold_dict[key][label].append((s[0], s[1]-1)) # 这个是原始的结果，原始结果中span的end都要减一
            else:
                gold_dict[key][label].append((span[0], span[1]-1))  # 这个是原始的结果，原始结果中span的end都要减一
    return gold_dict
if __name__ == "__main__":
    domain = "bn"
    new_data = []
    dic_data = {}
    # step 1: 使用os.listdir 获得所有的final数据
    folder_path = "annotated_final"  # 替换为实际的文件夹路径
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.endswith('final.json') and os.path.isfile(os.path.join(folder_path, f))]
    print(f"找到 {len(json_files)} 个以final.json结尾的文件:")
    for file_path in sorted(json_files):  # 排序以便有序处理
        print(f"  读取文件: {os.path.basename(file_path)}")
        data = read_json(file_path)
        print(f"数据个数为：{len(data['annotations'])}")
        new_data.extend(data['annotations'])
    print(f"\n总共读取了 {len(new_data)} 条数据")
    # dic_data = get_dic(new_data) # 需要保留语法错误和无论元的标签（无论元在评估的时候？？？怎么评估的来着） 这里可以获得语法错误和无论元的
    dic_data, grammar_error_num, no_arg, multi_arg = get_dic_ignore_unimportant(new_data) # 有语法错误的和无论元的标签
    

    # step 2: final数据加上所有gold里面正确的:全部正确的结果
    # org_data = get_all_orggold(read_json(f"final_data/test_{domain}_4llm_core_gold.conll"))
    org_data = get_all_orggold(read_json(f"llm/correct_data_{domain}_gold.json")) # 只有o1认为gold正确的才加入
    print(f'o1mini 认为gold正确的有：{len(org_data)}条')
    dic_data = merge_org_into_dic(dic_data, org_data)
    write_pickle(dic_data, "annotated_final/final_annotated_all.pkl") # 如果是nw的话，需要修改！！！
    print(f"合并后dic_data共 {len(dic_data)} 条")

    # step3: 构建gold查找表，评估小模型
    print("\n===== 基于人工标注后的评估 =====")
    gold_dict = build_gold_spans_from_dic(dic_data)
    print(f"gold_dict中谓词数: {len(gold_dict)}") # 去掉了谓词无论元和语法错误的句子
    # no_prd  = 0 
    # for key in gold_dict:
    #     if len(gold_dict[key])==0:
    #         no_prd += 0

    eval_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    print(f"评估数据共 {len(eval_data)} 个谓词") # 谓词数
    evaluate_model("semicrf",  "semicrf_label",  eval_data, gold_dict)
    evaluate_model("treecrf",  "treecrf_label",  eval_data, gold_dict)


    # 基于原始gold字段评估
    print("\n===== 基于原始gold评估 =====")
    gold_dict_origin = build_gold_dict_from_eval(eval_data)
    evaluate_model_gold("semicrf", "semicrf_label", eval_data, gold_dict_origin, gold_dict)
    evaluate_model_gold("treecrf", "treecrf_label", eval_data, gold_dict_origin, gold_dict)











