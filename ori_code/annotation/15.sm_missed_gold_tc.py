'''
bn中之前的可标可不标没有标注出来，需要重新找一下

如何获得可标可不标的数据呢？
先获得small model没有召回的corrected gold，然后对o1mini认为错误的用于标注；
'''
from config_data import read_json,write_json, write_pickle, read_pickle
import random
random.seed(42)
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
            # import pdb;pdb.set_trace()
            # 计算 semicrf 没有召回的 gold labels
            # 召回条件：label存在且span完全一致
            # import pdb;pdb.set_trace()
            # for label, span in corrected_args.items():
            #     import pdb;pdb.set_trace()
            semicrf_missed = {
                label for label, span in corrected_args.items()
                # if label not in semicrf or [(semicrf[label][0], semicrf[label][1] - 1)] != span
                if label not in semicrf
            }
        
            # 计算 treecrf 没有召回的 gold labels
            treecrf_missed = {
                label for label, span in corrected_args.items()
                # if label not in treecrf or [(treecrf[label][0], treecrf[label][1] - 1)] != span
                if label not in treecrf
            }
        
            # 求并集
            semicrf_missed_num += len(semicrf_missed) 
            treecrf_missed_num += len(treecrf_missed)
            all_missed = semicrf_missed | treecrf_missed
        
            # 构建结果字典
            for label in all_missed:
                key = (item[0], item[1], item[2], label)
                # import pdb;pdb.set_trace()
                result[key] = {
                    'sen': item[0],
                    'prd_word': item[1],
                    'pred_idx': item[2],
                    'prd.sense': dic_org[item]['prd.sense'],
                    'prd_lemma': dic_org[item]['prd_lemma'],
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
    for key, item in dic_data.items():
        sen, prd_word, prd_idx, label= key.split("\t")
        for span in item.get('selected_spans', []):
            gold_dict[(sen, prd_word, int(prd_idx))][label].append((span['start'], span['end']))
    return gold_dict

def build_annotation_data(results, llm_wrong_data, llmwrong_sm):
    """
    从大模型判断错误的数据中，找到在results中存在的条目，整理成标注格式。
    
    results: 上一步得到的字典，key=(sen, prd_word, pred_idx, label)
    llm_wrong_data: 大模型判断错误的json列表
    """
    annotation_list = []
    to_judge = []
    dic_llm_data = {}
    dic_llm_right_data = {}
    for item in llm_wrong_data:
        key = (item['sen'], item['prd_word'], item['prd_idx'], item['label'])
        dic_llm_data[key] = item

    for item in llmwrong_sm:
        key = (item['sen'], item['prd_word'], item['prd_idx'], item['label'])
        dic_llm_data[key] = item    
    
    
    for key in results:
        # 只保留在 results 中存在的条目
        if key not in dic_llm_data: # 需要保留，用于标注
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
        

    print(f'需要判断的个数为:{len(to_judge)}')
    return annotation_list, to_judge

def get_gold_index(data, frames, outfile_core):
    
    new_data_core = []
    new_data_all = []
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    idx = 0
    noframe = 0
    core_num , all_num = 0, 0
    for ins in data:
        key = ins[1]
        sen = key['sen']
        prd_word = key['prd_word']
        prd_lemma = key['prd_lemma'] 
        prd_sense = key['prd.sense']
        prd_idx = key['pred_idx']
        gold = key['corrected_gold']
        label = key['label']
        roles = []
        if prd_lemma in frames:
            senses = frames[prd_lemma]
            for ins in senses:
                if ins['role_set_id'] == ".".join([prd_lemma, prd_sense]):
                    roles = ins['roles']
        else:
            noframe += 1
    
        arg = gold[0]
        dic_core, dic_all = {}, {}
        idx += 1
        span = " ".join(sen.split()[arg[0]: arg[1]+1])
        dic_core["idx"] = idx
        dic_core['sen'] = sen
        dic_core['prd_word'] = prd_word
        dic_core['prd_lemma'] = prd_lemma
        dic_core['prd_sense'] = prd_sense
        dic_core['prd_idx']  = prd_idx
        dic_core['label'] = label
        dic_core['span'] = span
        dic_core['span_idx'] = [arg[0], arg[1]+1]
        
        if label in roles:
            # if sen == "A note scribbled by an officer on the Russian nuclear submarine Kursk has revealed that at least 23 of its 118 crewmembers survived the explosions which sank the vessel ." and prd_word=="survived":
            #         import pdb;pdb.set_trace()
            dic_core['span_mean'] =  roles[label]
            
        else:
            dic_core['span_mean'] = None
        if label in core_label:
            new_data_core.append(dic_core)
            core_num += 1
        new_data_all.append(dic_core)
        all_num += 1

    write_json(new_data_core, outfile_core)
    # write_json(new_data_all, outfile_all)
    print(f'No frame num in all predicate : {noframe}')
    print(f'Gold all predicate num :{all_num}, Gold core predicate num : {core_num}')

def get_annotation_instance(type_data, new_data, dic_alldata, choice_multi, choice_1, type=None):

    for instance in type_data:
        dic_new = {}
        sen = instance['sen']
        prd_word = instance['prd_word']
        prd_idx = instance['prd_idx']
        label = instance['label']
        span_mean = instance['span_mean']
        if "\t".join([sen, str(prd_idx)]) in dic_alldata:
            temp = dic_alldata["\t".join([sen, str(prd_idx)])]
            temp_gold = temp['gold']
            temp_semicrf = temp['semicrf_label']
            temp_treecrf = temp['treecrf_label']
            spans = defaultdict(list)
            if label in  temp_gold: # 和预测结果可能是相同label不同的spans
                gold_span = tuple(temp_gold[label])
                spans[gold_span].append("gold")
            if label in temp_semicrf:
                semicrf_span = tuple(temp_semicrf[label])
                spans[semicrf_span].append('semicrf')
            if label in temp_treecrf:
                treecrf_span = tuple(temp_treecrf[label])
                spans[treecrf_span].append('treecrf')
            if len(set(spans.keys()))>=2:
                choice_multi += 1
                options = spans
            else:
                choice_1 += 1
                options = spans
            dic_new['sen']  = sen
            dic_new['prd_word'] = prd_word
            dic_new['prd_idx']=prd_idx
            dic_new['label'] = label
            dic_new['span_mean'] = span_mean
            dic_new['options'] = options
            dic_new['type'] = type
            new_data.append(dic_new)
        else:
            print("error!找不到这个结果！")
    return new_data, choice_multi, choice_1
def get_anno(incorrect, smnotrecall, all_data, fileout, wheatherrandom=False, random_num=30):
    dic_incorrect = {}
    for item in incorrect:
        sen = item['sen']
        prd_word = item['prd_word']
        pred_idx = item['prd_idx']
        label = item['label']
        key = (sen, prd_word, pred_idx, label)
        dic_incorrect[key] = item
    
    print(len(dic_incorrect))

    need_to_anno = []
    for item in smnotrecall:
        sen = item['sen']
        prd_word = item['prd_word']
        pred_idx = item['prd_idx']
        label = item['label']
        key = (sen, prd_word, pred_idx, label)
        if key in dic_incorrect:
            # need_to_anno.append(dic_incorrect[key])  # 需要使用
            need_to_anno.append(item)  # 需要使用
            assert item['span_idx'] == dic_incorrect[key]['span_idx']
            # 
    new_data = []
    choice_multi, choice_1 = 0, 0
    dic_alldata = {"\t".join([i['sen'], str(i['pred.idx'])]):i for i in all_data}
    new_data, choice_multi, choice_1 = get_annotation_instance(need_to_anno, new_data, dic_alldata, choice_multi, choice_1, "the label of small model not recall")
    print(f'random 前{len(new_data)}')
    assert len(new_data) == len(need_to_anno)
    if wheatherrandom == True:
        new_data = random.sample(new_data, 30)

     # 根据sen排序,将相同的sen放在一起
    sorted_data = sorted(new_data, key=lambda x: (x['sen'], str(x['prd_idx'])))

    newest_data = []
    idx=0
    for instance in sorted_data:
        idx += 1
        new_instance= {'idx':idx} | instance
        newest_data.append(new_instance)
        sen = instance['sen']
        prd_word = instance['prd_word']
        pred_idx = instance['prd_idx']
        label = instance['label']
        key = (sen, prd_word, pred_idx, label)
        # if key == ('The federal government will not appeal the court ruling that cleared the way for same - sex unions .', 'cleared', 11, 'ARG1'):
        #         import pdb;pdb.set_trace()
    
    print(f'random 后{len(new_data)}')
    write_pickle(newest_data, fileout)
    return newest_data


def rm_30(final, data_30, fileout):
    new_data = []
    data_30_dic = {}
    for instance in data_30:
        sen = instance['sen']
        prd_word = instance['prd_word']
        pred_idx = instance['prd_idx']
        label = instance['label']
        key = (sen, prd_word, pred_idx, label)
        if key not in data_30_dic.keys():
            data_30_dic[key]= instance
    for instance in final:
        sen = instance['sen']
        prd_word = instance['prd_word']
        pred_idx = instance['prd_idx']
        label = instance['label']
        key = (sen, prd_word, pred_idx, label)
        if key not in data_30_dic.keys():
            new_data.append(instance)
    # 根据sen排序,将相同的sen放在一起
    sorted_data = sorted(new_data, key=lambda x: (x['sen'], str(x['prd_idx'])))

    newest_data = []
    idx=0
    for instance in sorted_data:
        idx += 1
        new_instance= {'idx':idx} | instance
        newest_data.append(new_instance)
    print(f'最终标注数据个数: {len(newest_data)}')
    write_pickle(newest_data, fileout)


if __name__=="__main__":
    domain = "tc"
    # step1 ：先找到小模型没有召回的corrected gold的并集

    # final corrected
    final_corrected =  read_json(f"analysis/test_{domain}_692_core_final.json")
    current_gold_data = build_gold_spans_from_dic(final_corrected)
    sm_data = read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    result = get_missed_gold_args(sm_data, current_gold_data)
    print(f"共找到 {len(result)} 条未被召回的 gold ARG0-ARG5 标签\n") #489


    # step2 ：需要大模型来判断下，对于大模型认为错误的选出来
    gold_bigmodel = read_json(f"llm/{domain}/incorrect_data_{domain}_gold.json")
    small_bigmodel = read_json(f"llm/{domain}/incorrect_data_{domain}_smallmodel.json")
    goldright_bigmodel = read_json(f"llm/{domain}/correct_data_{domain}_gold.json")

    annotation_data, to_judge = build_annotation_data(result, gold_bigmodel, small_bigmodel)
    # write_pickle(annotation_data, "anno/bn_annotation_sm_missed_gold_4optionalsupplement98.pkl")
    print(f"共找到 {len(annotation_data)} 条未被召回的 gold 且o1mini判断为错误的 ARG0-ARG5 标签\n")
    frames = read_json("/data/ljwang/span-SRL-LLM/propbank_frames_main/frame_out/frames_info_3.4.json") 
    file_out = f"final_data/{domain}/test_{domain}_4llm_core_smnotrecall_120.conll"
    get_gold_index(to_judge, frames, file_out)
    # 小模型没有召回的，gold正确，大模型判断正确

    # step3:  从大模型中获得错误的结果，目前错误的结果是包含了label在小模型但span不在的，目前我们要找label都没有召回的，是120个标签
    # 数据已保存到: llm/correct_data_bn_smnotrecall.json
    # 数据已保存到: llm/incorrect_data_bn_smnotrecall.json
    # correct: 247, incorrect: 34, all: 281

    # # 错误的一共21个
    # incorrect = read_json("llm/incorrect_data_bn_smnotrecall.json") # 这个end是+1后的
    # smnotrecall = read_json(file_out) # 这个end是+1后的
    # all_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.8.conll")
    # get_anno(incorrect, smnotrecall, all_data, fileout="anno/bn_smnotrecall_21.pkl")

    # # 从162-21=140个中正确的随机抽30个
    # correct = read_json("llm/correct_data_bn_smnotrecall.json") # 这个end是+1后的
    # smnotrecall = read_json(file_out) # 这个end是+1后的
    # all_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.8.conll")
    # get_anno(correct, smnotrecall, all_data, "anno/bn_smnotrecallright_random30.pkl", True, 30)
    
    # 从162-21=140个中正确的110
    # correct = read_json("llm/correct_data_bn_smnotrecall.json") # 这个end是+1后的
    # smnotrecall = read_json(file_out) # 这个end是+1后的
    # all_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.8.conll")
    # data_140 = get_anno(correct, smnotrecall, all_data, "anno/bn_smnotrecallright_random140.pkl")
    # pickle_30 = read_pickle(f"anno/bn_smnotrecallright_random30.pkl")
    # rm_30(data_140, pickle_30, f'anno/bn_smnotrecallright_110.pkl')
    