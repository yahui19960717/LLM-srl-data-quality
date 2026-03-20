from config_data import read_json, write_json, read_pickle, write_pickle, build_gold_spans_from_dic

'''
 可标可不标数据补充
 先看小模型没有召回的并集，如果大的话，再走大模型判断，选择错误的人工判别
'''
import json
from collections import defaultdict
def get_missed_gold_args(data, current_gold_data):
    """
    计算semicrf_label和treecrf_label各自没有召回的gold ARG0-ARG5标签，
    然后求并集，返回以(sen, prd_word, pred_idx, label)为key的字典。
    """
    target_labels = {f'ARG{i}' for i in range(6)}  # ARG0-ARG5
    dic_org = {}
    result = {}
    semicrf_missed_num, treecrf_missed_num = 0, 0

    for item in data:
        sen = item['sen']
        prd_word = item['prd_word']
        pred_idx = item['pred.idx']
        # gold = item['gold']
        # semicrf = item['semicrf_label']
        # treecrf = item['treecrf_label']
        key = (sen, prd_word, pred_idx)
        dic_org[key]  = item
        
    for item in current_gold_data:
        if item in dic_org:
            gold = dic_org[item]['gold']
            semicrf = dic_org[item]['semicrf_label']
            treecrf = dic_org[item]['treecrf_label']
            # 只保留 ARG0-ARG5 的 gold 标签
            gold_args = {label: span for label, span in gold.items() if label in target_labels}
            corrected_args = current_gold_data[item]
            # 计算 semicrf 没有召回的 gold labels
            # 召回条件：label存在且span完全一致
            # import pdb;pdb.set_trace()
            # for label, span in corrected_args.items():
            #     import pdb;pdb.set_trace()
            semicrf_missed = {
                label for label, span in corrected_args.items()
                if label not in semicrf or [(semicrf[label][0], semicrf[label][1] - 1)] != span
            }
        
            # 计算 treecrf 没有召回的 gold labels
            treecrf_missed = {
                label for label, span in corrected_args.items()
                if label not in treecrf or [(treecrf[label][0], treecrf[label][1] - 1)] != span
            }
        
            # 求并集
            semicrf_missed_num += len(semicrf_missed) 
            treecrf_missed_num += len(treecrf_missed)
            all_missed = semicrf_missed | treecrf_missed
        
            # 构建结果字典
            for label in all_missed:
                key = (item[0], item[1], item[2], label)
                result[key] = {
                    'sen': item[0],
                    'prd_word': item[1],
                    'pred_idx': item[2],
                    'label': label,
                    'corrected_gold':corrected_args[label] ,
                    'semicrf_span': semicrf.get(label),
                    'treecrf_span': treecrf.get(label),
                    'missed_by': {
                        'semicrf': label in semicrf_missed,
                        'treecrf': label in treecrf_missed,
                    }
                }
    print(f"  semicrf_missed   : {semicrf_missed_num}")
    print(f"  treecrf_missed   : {treecrf_missed_num}")
    
    return result

def build_annotation_data(results, llm_wrong_data, llmwrong_sm):
    """
    从大模型判断错误的数据中，找到在results中存在的条目，整理成标注格式。
    
    results: 上一步得到的字典，key=(sen, prd_word, pred_idx, label)
    llm_wrong_data: 大模型判断错误的json列表
    """
    annotation_list = []
    to_judge = []
    dic_llm_data = {}
    for item in llm_wrong_data:
        key = (item['sen'], item['prd_word'], item['prd_idx'], item['label'])
        dic_llm_data[key] = item

    for item in llmwrong_sm:
        key = (item['sen'], item['prd_word'], item['prd_idx'], item['label'])
        dic_llm_data[key] = item    
    
    for key in results:
        # 只保留在 results 中存在的条目
        if key not in dic_llm_data:
            to_judge.append([key, results[key]])
            continue
        

        # options: {span_idx_tuple: ['gold']}
        options = defaultdict(list)
        options[tuple(item['span_idx'])].append('gold')

        annotation_list.append({
            'idx': item['idx'],
            'sen': item['sen'],
            'prd_word': item['prd_word'],
            'prd_idx': item['prd_idx'],
            'label': item['label'],
            'span_mean': item['span_mean'],
            'options': options,
            'type': 'gold right, o1mini wrong'
        })
        

    print(len(to_judge))
    return annotation_list


if __name__=="__main__":
    domain = "bn"

    # step1 ：先找到小模型没有召回的gold的并集
    org_gold_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    final_data = read_pickle("annotated_final/final_annotated_all.pkl") # final annotated data bn
    current_gold_data = build_gold_spans_from_dic(final_data)
    result = get_missed_gold_args(org_gold_data, current_gold_data)
    print(f"共找到 {len(result)} 条未被召回的 gold ARG0-ARG5 标签\n") #489
    # for key, val in result.items():
    #     print(f"Key: {key}")
    #     print(f"  gold_span   : {val['gold_span']}")
    #     print(f"  semicrf_span: {val['semicrf_span']}  missed={val['missed_by']['semicrf']}")
    #     print(f"  treecrf_span: {val['treecrf_span']}  missed={val['missed_by']['treecrf']}")
    #     print()
    
    
    # step2 ：需要大模型来判断下，对于大模型认为错误的选出来
    gold_bigmodel = read_json(f"llm/incorrect_data_{domain}_gold.json")
    small_bigmodel = read_json(f"llm/incorrect_data_{domain}_smallmodel.json")

    annotation_data = build_annotation_data(result, gold_bigmodel, small_bigmodel)
    # write_pickle(annotation_data, "anno/bn_annotation_sm_missed_gold_4optionalsupplement98.pkl")
    print(f"共找到 {len(annotation_data)} 条未被召回的 gold 且o1mini判断为错误的 ARG0-ARG5 标签\n")