import json
from collections import defaultdict
import pickle
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

def read_pickle(file):
    """从文件读取pickle数据"""
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(f'read pickle :data number {len(data)}')
    return data

def parse_response(instance, response):
    try:
        parse_tag = False
        res_str = response.strip()
        if not res_str:
            print("Error:empty response!")
            return response, parse_tag
        if res_str.startswith("```json"):
            res_str = res_str[7:]
        if res_str.endswith("```"):
            res_str = res_str[:-3]
        res_str = res_str.strip()

        res_data = json.loads(res_str)
        parse_tag = True
        res_data['idx'] = instance['idx']
        res_data['sen'] = instance['sen']
        res_data['prd_word'] = instance['prd_word']
        res_data['span'] = instance['span']
        res_data['prd_lemma'] = instance['prd_lemma']
        res_data['prd_sense'] = instance['prd_sense']
        res_data['prd_idx'] = instance['prd_idx']
        res_data['label'] = instance['label']
        res_data['span_idx'] = instance['span_idx']
        return res_data, parse_tag
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag
    except Exception as e:
        print(f"其他错误: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag
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
            # 该谓词未在dic_data中，跳过 # 包含了之前处理的be动词、无论元的、语法错误的等等
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

def build_gold_spans_from_dic(dic_data): # 处理最终标注的结果
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


def build_gold_dict_from_eval(eval_data): # 处理gold
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