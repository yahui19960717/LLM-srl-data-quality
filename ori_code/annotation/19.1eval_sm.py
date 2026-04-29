"""
那我们可以先使用这500测试集：原来的标注及现在修正后的标注，重新评估一下两个小模型的结果，只看核心角色，看一下精确率、召回率和F1的变化

1. 目前我需要把所有的标注数据都放在一起作为最终的标注结果(current gold)（500句）

2. 获得原来的semicrf预测和treecrf预测(注意，这里的semicrf和treecrf都是带be动词的)

3. 写评估脚本：目前预测的PRF值 (current gold) :注意，这里的评估需要更改，只要有一个符合的就算对
    对于可标可不标的无论预测不预测都算对

4. 写评估脚本：原来预测的PRF值（previous gold)


"""
import os
import json
import pickle
from collections import defaultdict
from config_data import read_json

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

def evaluate_model(model_name, pred_field, eval_data, gold_dict, other_info):
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

        # gold_set = spans_to_set(gold_spans)
        
        temp_sen, temp_word, temp_idx = lookup_key
        
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
            

            # 先判断是不是语法错误
            grammar = other_info[(temp_sen, temp_word, temp_idx)][lbl].get('grammar_error_desc', None) if isinstance(other_info[(temp_sen, temp_word, temp_idx)][lbl], dict) else []
            opt = other_info[(temp_sen, temp_word, temp_idx)][lbl].get('optional', None) if isinstance(other_info[(temp_sen, temp_word, temp_idx)][lbl], dict) else []
            if grammar == "有语法错误":
                continue

            if pred_list:
                # 判断是否命中（复用原有的单/多span逻辑）
                if len(pred_list) == 1:
                    hit = pred_list[0] in gold_set
                else:
                    hit = any(pred_span in gold_set for pred_span in pred_list)

                if hit:
                    overall['tp'] += 1
                    per_label[lbl]['tp'] += 1
                else:
                    # 预测了但错了 → FP 一定计
                    overall['fp'] += 1
                    per_label[lbl]['fp'] += 1
                    # FN 只有非 optional 且 gold 有时才计
                    if lbl in gold_labels and not opt:
                        overall['fn'] += 1
                        per_label[lbl]['fn'] += 1
            else:
                # 没有预测，gold 有 → FN，但 optional 跳过
                if lbl in gold_labels and not opt:
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
    P, R ,F = p,r,f
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
    return P, R ,F 
    

def evaluate_model_gold(model_name, pred_field, eval_data, org_dict):
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
        
        if lookup_key not in org_dict:
            # 该谓词未在dic_data中，跳过 # 包含了之前处理的be动词、只有be动词的谓词等等
            unmatched_pred += 1
            continue
        matched += 1

        gold_spans = org_dict[lookup_key]   # {label: [(start,end),...]}
        pred_spans = item.get(pred_field, {}) # {label: [start, end]}  注意：单个span是列表[start,end]

        # gold_set = spans_to_set(gold_spans)
        
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
                else: # 如果预测多个span就遍历
                    hit = False
                    for pred_span in pred_list:
                        if pred_span in gold_set:
                            # 命中gold中任意一个span → TP
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
    P, R ,F = p,r,f
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
    return P, R, F

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

if __name__ == "__main__":
    # ## tc评估
    # domain = "tc"
    # print(domain)
    # new_data = []
    # dic_data = {}
    # # step 1: 获得评估数据
    # eval_data = read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    # print(f"评估数据共 {len(eval_data)} 个谓词") # 谓词数
    
    # # step 2: 获得所有的final数据
    # # corrected data
    # print("\n===== 基于人工标注后的评估 =====")
    # final_data = read_json(f"analysis/test_{domain}_core_final_v2.json")
    # corrected_data, other_info = build_corrected_data_for_dic(final_data)
    # print(len(eval_data), len(corrected_data))
    # P1, R1, F1 = evaluate_model("semicrf",  "semicrf_label",  eval_data, corrected_data, other_info)
    # P2, R2, F2 = evaluate_model("treecrf",  "treecrf_label",  eval_data, corrected_data, other_info)
    

    # # # 基于原始gold字段评估
    # print("\n===== 基于原始gold评估 =====")
    # gold_dict_origin = build_gold_dict_from_eval(eval_data)
    # P3, R3, F3 = evaluate_model_gold("semicrf", "semicrf_label", eval_data, gold_dict_origin)
    # P4, R4, F4 = evaluate_model_gold("treecrf", "treecrf_label", eval_data, gold_dict_origin)

    # print(f'SemiCRF-corrected: {P1*100:.2f}, {R1*100:.2f}, {F1*100:.2f}')
    # print(f'SemiCRF-org: {P3*100:.2f}, {R3*100:.2f}, {F3*100:.2f}')

    # print(f'TreeCRF-corrected: {P2*100:.2f}, {R2*100:.2f}, {F2*100:.2f}')
    # print(f'TreeCRF-org: {P4*100:.2f}, {R4*100:.2f}, {F4*100:.2f}')

    # print(f'SemiCRF diff: P:{(P1-P3)*100:+.2f}% R:{(R1-R3)*100:+.2f}% F:{(F1-F3)*100:+.2f}%')
    # print(f'TreeCRF diff: P:{(P2-P4)*100:+.2f}% R:{(R2-R4)*100:+.2f}% F:{(F2-F4)*100:+.2f}%')
    # exit()


    ### bn评估
    domain = "bn"
    new_data = []
    dic_data = {}
    # step 1: 获得评估数据
    eval_data = read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    print(f"评估数据共 {len(eval_data)} 个谓词") # 谓词数
    
    # step 2: 获得所有的final数据
    # corrected data
    print("\n===== 基于人工标注后的评估 =====")
    final_data = read_json(f"analysis/test_bn_500_core_final_v4.json")
    corrected_data, other_info = build_corrected_data_for_dic(final_data)
    print(len(eval_data), len(corrected_data))
    P1, R1, F1 = evaluate_model("semicrf",  "semicrf_label",  eval_data, corrected_data, other_info)
    P2, R2, F2 = evaluate_model("treecrf",  "treecrf_label",  eval_data, corrected_data, other_info)
    

    # # 基于原始gold字段评估
    print("\n===== 基于原始gold评估 =====")
    gold_dict_origin = build_gold_dict_from_eval(eval_data)
    P3, R3, F3 = evaluate_model_gold("semicrf", "semicrf_label", eval_data, gold_dict_origin)
    P4, R4, F4 = evaluate_model_gold("treecrf", "treecrf_label", eval_data, gold_dict_origin)

    print(f'SemiCRF-corrected: {P1*100:.2f}, {R1*100:.2f}, {F1*100:.2f}')
    print(f'SemiCRF-org: {P3*100:.2f}, {R3*100:.2f}, {F3*100:.2f}')

    print(f'TreeCRF-corrected: {P2*100:.2f}, {R2*100:.2f}, {F2*100:.2f}')
    print(f'TreeCRF-org: {P4*100:.2f}, {R4*100:.2f}, {F4*100:.2f}')

    print(f'SemiCRF diff: P:{(P1-P3)*100:+.2f}% R:{(R1-R3)*100:+.2f}% F:{(F1-F3)*100:+.2f}%')
    print(f'TreeCRF diff: P:{(P2-P4)*100:+.2f}% R:{(R2-R4)*100:+.2f}% F:{(F2-F4)*100:+.2f}%')
    exit()










