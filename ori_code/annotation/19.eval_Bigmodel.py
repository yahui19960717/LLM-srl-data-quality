'''
 我感觉可以把大模型结果（o1mini)再统计一下；
 1. 先获得所有大模型的预测结果(正确和错误）：目前用大模型先判断了所有的gold结果+ 多个小模型预测结果（不与golden匹配的）

 2. 利用final人工标注之后的数据来评估大模型预测结果中正确的
'''
import json
from collections import defaultdict
import pickle
from config_data import read_json, write_json, write_pickle, read_pickle, compute_prf, spans_to_set, evaluate_model
def get_variable_name(var, namespace=None):
    if namespace is None:
        namespace = globals()
    return [name for name, value in namespace.items() if value is var]

def build_bigmodel_dict_from_eval(data_list):
    """
    直接从eval_data中的gold字段构建bigmodel_dict
    格式: {(sen, prd_word, prd_idx): {label: [(start, end)]}}

    gold_bigmodel, pred_bigmodel, bm_bigmodle
    """
    bigmodel_all = []
    for data in data_list:
        print(f"Big model judge right number for {get_variable_name(data)}: {len(data)}")
        bigmodel_all += data
    print(len(bigmodel_all))
    # print(f"Big model judge right number for gold: {len(gold_bigmodel)}")
    # print(f"Big model judge right number for pred: {len(pred_bigmodel)}")
    # bigmodel_all = gold_bigmodel + pred_bigmodel
    bigmodel_dict = defaultdict(lambda: defaultdict(list))
    for item in bigmodel_all:
        sen     = item['sen']
        prd_word = item['prd_word']
        prd_idx  = item.get('pred.idx', item.get('prd_idx', -1))
        label = item.get("label", None)
        span = item.get("span_idx", None)
        key = (sen, prd_word, prd_idx)
        bigmodel_dict[key][label].append((span[0], span[1]-1))  # 这个是原始的结果，原始结果中span的end都要减一
    return bigmodel_dict

def evaluate_model_basedon_correcteddata(model_name, eval_data, gold_dict, other_info):
    """
    根据dic_data构建的gold_dict评估某个小模型的ARG0-ARG5 PRF。
    
    eval_data: bigmodel 所有预测结果为correct的例子
    current_data: build_gold_spans_from_dic的输出
    """
    target_labels = {f'ARG{i}' for i in range(6)}
    # 整体计数
    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    # 每个标签计数
    per_label = {lbl: {'tp': 0, 'fp': 0, 'fn': 0} for lbl in target_labels}

    matched = 0
    unmatched_pred = 0
    for lookup_key in eval_data: # 谓词级别的处理
        
        if lookup_key not in gold_dict:
            # 该谓词未在dic_data中，跳过 # 包含了之前处理的be动词、无论元的、语法错误的等等
            unmatched_pred += 1
            print(lookup_key)
            continue
        matched += 1 # 谓词匹配的
        gold_spans = gold_dict[lookup_key]   # {label: [(start,end),...]} 对于某个谓词所有的label
        pred_spans=eval_data[lookup_key] #注意：单个span是{label: [(start,end)]}

        gold_set = spans_to_set(gold_spans)
        temp_sen, temp_word, temp_idx = lookup_key
        
        # 将pred转成 {label: (start, end)}，只处理ARG0-ARG5
        pred_dict = {}
        for label, span in pred_spans.items():  # 这里的span可能有多个
            if label not in target_labels:
                continue
            pred_dict[label] = span
        # 以label为单位计算TP/FP/FN
        # gold中出现的label集合 与 pred中出现的label集合
        gold_labels = {lbl for lbl in gold_spans if lbl in target_labels} # 获得所有的label
        pred_labels = set(pred_dict.keys())
        all_labels  = gold_labels | pred_labels # 所有可能的labels
        for lbl in all_labels:
            gold_set = set(gold_spans.get(lbl, []))  # 该label的所有合法span
            pred_list = pred_dict.get(lbl, [])        # 该label的预测span（最多一个）
            # 先判断是不是语法错误
            grammar = other_info[(temp_sen, temp_word, temp_idx)][lbl].get('grammar_error_desc', None) if isinstance(other_info[(temp_sen, temp_word, temp_idx)][lbl], dict) else []
            opt = other_info[(temp_sen, temp_word, temp_idx)][lbl].get('optional', None) if isinstance(other_info[(temp_sen, temp_word, temp_idx)][lbl], dict) else []
            if grammar == "有语法错误":
                continue
            elif opt==True:
                overall['tp'] += 1
                per_label[lbl]['tp'] += 1
                overall['fn'] += 1
                per_label[lbl]['fn'] += 1

                continue

            if pred_list:
                if len(pred_list) == 1: # 如果只有1个span
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
                    hit = False
                    for pred_span in pred_list:
                        if pred_span in gold_set:
                            # 命中gold中任意一个span → TP
                            overall['tp'] += 1
                            per_label[lbl]['tp'] += 1
                            hit = True
                            break
                    if hit:
                        overall['tp'] += 1
                        per_label[lbl]['tp'] += 1
                    else:
                        # 整个预测列表没有命中任何真实span
                        overall['fp'] += 1   # 或者根据需求：整体预测错误
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
    print(f"{'='*75}")

def evaluate_model_v3(model_name, eval_data, gold_dict):
    """
    根据dic_data构建的gold_dict评估某个小模型的ARG0-ARG5 PRF。
    
    eval_data: bigmodel 所有预测结果为correct的例子
    current_data: build_gold_spans_from_dic的输出
    """
    target_labels = {f'ARG{i}' for i in range(6)}
    # 整体计数
    overall = {'tp': 0, 'fp': 0, 'fn': 0}
    # 每个标签计数
    per_label = {lbl: {'tp': 0, 'fp': 0, 'fn': 0} for lbl in target_labels}

    matched = 0
    unmatched_pred = 0
    for lookup_key in eval_data: # 谓词级别的处理
        
        if lookup_key not in gold_dict:
            # 该谓词未在dic_data中，跳过 # 包含了之前处理的be动词、无论元的、语法错误的等等
            unmatched_pred += 1
            # print(lookup_key)
            continue
        matched += 1 # 谓词匹配的
        gold_spans = gold_dict[lookup_key]   # {label: [(start,end),...]} 对于某个谓词所有的label
        pred_spans=eval_data[lookup_key] #注意：单个span是{label: [(start,end)]}

        gold_set = spans_to_set(gold_spans)
        temp_sen, temp_word, temp_idx = lookup_key
        
        # 将pred转成 {label: (start, end)}，只处理ARG0-ARG5
        pred_dict = {}
        for label, span in pred_spans.items(): 
            if label not in target_labels:
                continue
            pred_dict[label] = [(span[0][0], span[0][1])] 

        # 以label为单位计算TP/FP/FN
        # gold中出现的label集合 与 pred中出现的label集合
        gold_labels = {lbl for lbl in gold_spans if lbl in target_labels} # 获得所有的label
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
    print(f"{'='*75}")

def build_corrected_data_for_dic(dict_data):
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
    corrected_dict = defaultdict(lambda: defaultdict(list))
    other_info = defaultdict(lambda: defaultdict(list)) 
    for ins in dict_data:
        sen, prd_word, prd_idx, label = ins.split("\t")
        item = dict_data[ins]
        for span in item.get('selected_spans', []): # 若无span就不记录了
            corrected_dict[(sen, prd_word, int(prd_idx))][label].append((span['start'], span['end']))
            other_info[(sen, prd_word, int(prd_idx))][label] = item
    print(len(corrected_dict), len(other_info))
    return corrected_dict, other_info
        
    # for (sen, prd_word, prd_idx, label), item in dic_data.items():
    #     for span in item.get('selected_spans', []):
    #         gold_dict[(sen, prd_word, prd_idx)][label].append((span['start'], span['end']))
    # return corrected_dict

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

def evaluate_accuracy(model_name, eval_data, gold_dict, other_info):
    """
    计算准确率：正确命中的 (谓词, label) 对数 / gold 中所有有效 (谓词, label) 对数。

    对于每个 (谓词, label) 对：
    - 语法错误 → 跳过（不计入分母）
    - optional → 直接算正确（计入分子和分母）
    - 正常情况 → 预测 span 列表中任意一个命中 gold span 集合即算正确
    """
    target_labels = {f'ARG{i}' for i in range(6)}
    overall_correct = 0
    overall_total = 0
    grammar_error, optional_num = 0, 0
    per_label = {lbl: {'correct': 0, 'total': 0} for lbl in target_labels}
    label_num = 0
    for lookup_key, gold_spans in gold_dict.items():
        temp_sen, temp_word, temp_idx = lookup_key
        pred_spans = eval_data.get(lookup_key, {})

        for lbl in gold_spans:
            label_num += 1
            if lbl not in target_labels:
                continue
            
            # 获取 other_info
            info = other_info.get(lookup_key, {}).get(lbl, {})
            if isinstance(info, dict):
                grammar = info.get('grammar_status', None)
                opt = info.get('optional', None)
            else:
                grammar = None
                opt = None

            # 语法错误 → 跳过
            if grammar == "有语法错误":
                grammar_error += 1
                continue

            # 计入分母
            overall_total += 1
            per_label[lbl]['total'] += 1

            # optional → 有预测则正常判断，没有预测则跳过（不计入分母）
            if opt is True:
                optional_num += 1
                pred_list = pred_spans.get(lbl, [])
                if not pred_list:
                    # 没有预测，跳过，不计入分母也不计入分子
                    overall_total -= 1          # 刚才已经 +1 了，撤回
                    per_label[lbl]['total'] -= 1
                    continue
                # 有预测，正常判断对不对
                gold_set = set(gold_spans.get(lbl, []))
                if any(span in gold_set for span in pred_list):
                    overall_correct += 1
                    per_label[lbl]['correct'] += 1
                # 预测错了：计入分母但不计入分子（即算错）
                continue

            # 正常判断：预测中任意一个 span 命中 gold 中任意一个 span
            gold_set = set(gold_spans.get(lbl, []))
            pred_list = pred_spans.get(lbl, [])
            if pred_list and any(span in gold_set for span in pred_list):
                overall_correct += 1
                per_label[lbl]['correct'] += 1

    # 输出
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0.0
    print(f"\n{'='*65}")
    print(f"模型: {model_name}  |  Accuracy 评估")
    print(f"{'='*65}")
    print(f"{'标签':<12} {'Accuracy(%)':>12} {'Correct':>9} {'Total':>9}")
    print(f"{'-'*65}")
    print(f"{'Overall':<12} {overall_acc:>12.2f} {overall_correct:>9} {overall_total:>9}")
    print(f"{'-'*65}")

    for lbl in sorted(target_labels):
        c = per_label[lbl]
        if c['total'] == 0:
            continue
        acc = c['correct'] / c['total'] * 100
        print(f"{lbl:<12} {acc:>12.2f} {c['correct']:>9} {c['total']:>9}")
    print(f"{'='*65}")
    print(f"语法错误的个数为:{grammar_error}, 可标可不标的个数为: {optional_num}")
    print(f'评估数据的谓词个数为: {len(eval_data)}, corrected数据的谓词个数：{len(gold_dict)}')
    # import pdb;pdb.set_trace()
    # assert label_num == overall_total+grammar_error 
    return overall_correct, overall_total, per_label

def evaluate_accuracy_orggold(model_name, eval_data, gold_dict):
    """
    计算准确率：正确命中的 (谓词, label) 对数 / gold 中所有有效 (谓词, label) 对数。

    对于每个 (谓词, label) 对：
    - 语法错误 → 跳过（不计入分母）
    - optional → 直接算正确（计入分子和分母）
    - 正常情况 → 预测 span 列表中任意一个命中 gold span 集合即算正确
    """
    target_labels = {f'ARG{i}' for i in range(6)}
    overall_correct = 0
    overall_total = 0
    per_label = {lbl: {'correct': 0, 'total': 0} for lbl in target_labels}

    for lookup_key, gold_spans in gold_dict.items():
        temp_sen, temp_word, temp_idx = lookup_key
            
        pred_spans = eval_data.get(lookup_key, {})
        for lbl in gold_spans:
            if lbl not in target_labels:
                continue
            
            overall_total += 1
            per_label[lbl]['total'] += 1
            # 正常判断：预测中任意一个 span 命中 gold 中任意一个 span
            gold_set = set(gold_spans.get(lbl, []))
            pred_list = pred_spans.get(lbl, [])
            if pred_list and any(span in gold_set for span in pred_list):
                overall_correct += 1
                per_label[lbl]['correct'] += 1

    # 输出
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0.0
    print(f"\n{'='*65}")
    print(f"模型: {model_name}  |  Accuracy 评估")
    print(f"{'='*65}")
    print(f"{'标签':<12} {'Accuracy(%)':>12} {'Correct':>9} {'Total':>9}")
    print(f"{'-'*65}")
    print(f"{'Overall':<12} {overall_acc:>12.2f} {overall_correct:>9} {overall_total:>9}")
    print(f"{'-'*65}")

    for lbl in sorted(target_labels):
        c = per_label[lbl]
        if c['total'] == 0:
            continue
        acc = c['correct'] / c['total'] * 100
        print(f"{lbl:<12} {acc:>12.2f} {c['correct']:>9} {c['total']:>9}")
    print(f"{'='*65}")

    return overall_correct, overall_total, per_label


if __name__ == "__main__":
    domain = "tc"
    gold_bigmodel = read_json(f"llm/{domain}/correct_data_{domain}_gold.json")
    pred_bigmodel = read_json(f"llm/{domain}/correct_data_{domain}_smallmodel.json")
    bm_added = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/llm/tc/correct_data_{domain}_smnotrecall_filter.json")
    list_data = [gold_bigmodel, pred_bigmodel, bm_added]
    eval_data = build_bigmodel_dict_from_eval(list_data) # gold+pred+bmtojudgepostprocessing （大模型后续判断的）

    # corrected data
    final_data = read_json(f"analysis/test_{domain}_core_final_v2.json")
    corrected_data, other_info = build_corrected_data_for_dic(final_data)
    print(len(eval_data), len(corrected_data))
    evaluate_accuracy("o1mini", eval_data, corrected_data, other_info) # 这里的gold是不需要减1的

    # 基于原始gold字段评估
    print("\n===== 基于原始gold评估 =====")
    temp_data = read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    gold_dict_origin = build_gold_dict_from_eval(temp_data) # 这里的gold是需要减1的
    
    evaluate_accuracy_orggold("o1mini in org gold", eval_data, gold_dict_origin)
    
    exit()


    # step1 : 获得所有o1mini判断的结果：
    domain = "bn"
    gold_bigmodel = read_json(f"llm/correct_data_{domain}_gold.json")
    pred_bigmodel = read_json(f"llm/correct_data_{domain}_smallmodel.json")
    bm_added = read_json(f"llm/correct_data_{domain}_bmnotjudgedl.json")
    list_data = [gold_bigmodel, pred_bigmodel, bm_added]
    eval_data = build_bigmodel_dict_from_eval(list_data) # gold+pred+bmtojudgepostprocessing （大模型后续判断的）

    # corrected data
    final_data = read_json(f"analysis/test_bn_500_core_final_v4.json")
    corrected_data, other_info = build_corrected_data_for_dic(final_data)
    print(len(eval_data), len(corrected_data))
    evaluate_accuracy("o1mini", eval_data, corrected_data, other_info) # 这里的gold是不需要减1的

    # 基于原始gold字段评估
    print("\n===== 基于原始gold评估 =====")
    temp_data = read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    gold_dict_origin = build_gold_dict_from_eval(temp_data) # 这里的gold是需要减1的
    
    evaluate_accuracy_orggold("o1mini in org gold", eval_data, gold_dict_origin)


    
