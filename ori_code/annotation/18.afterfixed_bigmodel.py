'''
 我感觉可以把大模型结果（o1mini)再统计一下；
 1. 先获得所有大模型的预测结果(正确和错误）：目前用大模型先判断了所有的gold结果+ 多个小模型预测结果（不与golden匹配的）

 2. 利用final人工标注之后的数据来评估大模型预测结果中正确的
'''
import json
from collections import defaultdict
import pickle
from config_data import read_json, write_json, write_pickle, read_pickle, compute_prf, spans_to_set, evaluate_model
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


def build_bigmodel_dict_from_eval(gold_bigmodel, pred_bigmodel):
    """
    直接从eval_data中的gold字段构建bigmodel_dict
    格式: {(sen, prd_word, prd_idx): {label: [(start, end)]}}
    """
    print(f"Big model judge right number for gold: {len(gold_bigmodel)}")
    print(f"Big model judge right number for pred: {len(pred_bigmodel)}")
    bigmodel_all = gold_bigmodel + pred_bigmodel
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

def build_bigmodel_dict_from_eval_single(gold_bigmodel):
    """
    直接从eval_data中的gold字段构建bigmodel_dict
    格式: {(sen, prd_word, prd_idx): {label: [(start, end)]}}
    """
    print(f"Big model judge right number for gold: {len(gold_bigmodel)}")
    # print(f"Big model judge right number for pred: {len(pred_bigmodel)}")
    bigmodel_all = gold_bigmodel
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
    

def evaluate_model_v2(model_name, eval_data, gold_dict):
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
        matched += 1

        gold_spans = gold_dict[lookup_key]   # {label: [(start,end),...]}
        pred_spans=eval_data[lookup_key] #注意：单个span是{label: [(start,end)]}
        gold_set = spans_to_set(gold_spans)
        
        # 将pred转成 {label: (start, end)}，只处理ARG0-ARG5
        pred_dict = {}
        for label, span in pred_spans.items(): 
            if label not in target_labels:
                continue
            pred_dict[label] = [(span[0][0], span[0][1])] 

        # 以label为单位计算TP/FP/FN
        # gold中出现的label集合 与 pred中出现的label集合
        gold_labels = {lbl for lbl in gold_spans if lbl in target_labels}
        pred_labels = set(pred_dict.keys())
        # import pdb;pdb.set_trace()
        all_labels  = gold_labels | pred_labels # 所有可能的labels
        # import pdb;pdb.set_trace()
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

    # step1 : 获得所有大模型判断的结果：
    domain = "bn"
    gold_bigmodel = read_json(f"llm/correct_data_{domain}_gold.json")
    pred_bigmodel = read_json(f"llm/correct_data_{domain}_smallmodel.json")
    eval_data = build_bigmodel_dict_from_eval(gold_bigmodel, pred_bigmodel) # gold+pred

    # eval_data = build_bigmodel_dict_from_eval_single(gold_bigmodel) # 仅gold结果
    final_data = read_pickle("annotated_final/final_annotated_all.pkl") # final annotated data bn
    gold_data = build_gold_spans_from_dic(final_data)
    evaluate_model_v2("o1mini", eval_data, gold_data) # 这里的gold是不需要减1的

    # 基于原始gold字段评估
    print("\n===== 基于原始gold评估 =====")
    temp_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    gold_dict_origin = build_gold_dict_from_eval(temp_data) # 这里的gold是需要减1的
    
    evaluate_model_v2("o1mini in org gold", eval_data, gold_dict_origin)


