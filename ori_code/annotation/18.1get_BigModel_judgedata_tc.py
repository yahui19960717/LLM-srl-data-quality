from config_data import read_json, write_json

'''
为了评估大模型判断的性能，我们首先需要对corrected golden中所有span让大模型判断的结果
先看一下人工修正的有多少，我建议以新golden中的label+span为核心，在跑过o1 mini的数据中看哪些已经跑过了（原始golden和小模型产出，跑o1的结果，正确和错误都含有的），剩下没有跑过的span，再用大模型跑一下
'''
target_labels = {f'ARG{i}' for i in range(6)}

def get_variable_name(var, namespace=None):
    if namespace is None:
        namespace = globals()
    return [name for name, value in namespace.items() if value is var]

def get_corrected_dic(data):
    """
    根据final_data构建final corrected golden dict :ARG0-ARG5 PRF。
    输入：key:sen\tprd_word\tprd_idx\tlabel， item是标注结果
    输出：key:(sen, prd_word, prd_idx, label, (start, end)),item还是之前的
    """
    dic_data = {}
    span_num, grammar_num, optional_num = 0, 0, 0
    for key in data:
        sen, prd_word, prd_idx, label = key.split("\t")
        prd_idx = int(prd_idx)
        ins = data[key]
        grammar = ins.get('grammar_status', None)
        optional = ins.get('optional', None)
        spans = ins.get('selected_spans', [])

        # 只看grammar正确的结果 ins[] == "有语法错误"
        if grammar == "有语法错误":
            grammar_num += 1
            continue
        # 去掉可标可不标的结果，无论大模型如何判断，都算是对的
        elif optional == True: # 可标可不标
            optional_num += 1
            continue
        else:
            for span in spans: # 遍历所有的span，因为需要将(sen,prd_word,prd_idx,label,span)作为key
                start, end, text = span['start'], span['end'], span['text']
                temp_key = (sen, prd_word, prd_idx, label, (start, end+1)) #正确的span +1之后和原来的契合
                dic_data[temp_key] = ins
                span_num += 1
    
    assert len(dic_data) == span_num
    print(f'span-level的instance个数为: {len(dic_data)}')
    print(f'grammar error: {grammar_num}, 可标可不标的数据: {optional_num}')

    return dic_data

def get_sm_judged_dic(list_data):
    """
    根据list_data构建大模型已经判断的数据的dict
    输入：每个data是一个list，每个list的元素是一个词典 
    输出：key:(sen, prd_word, prd_idx, label, (start, end)),item还是之前的list元素
    """
    temp_n = len(list_data)
    dic_judged_data = {}
    all_data = 0
    print(f'共有{temp_n}个大模型判断过的数据')
    for i in range(temp_n):
        num = 0
        data = list_data[i] 
        for ins in data:
            sen, prd_word, prd_idx, label, span = ins['sen'], ins['prd_word'], ins['prd_idx'], ins['label'], (ins['span_idx'][0], ins['span_idx'][1]) #
            key = (sen, prd_word, prd_idx, label, span)
            dic_judged_data[key] = ins
            num += 1
        all_data += num
        print(f'{get_variable_name(data)}数据的个数为: {num}')
    print(f"全部判断过的数据个数为: {all_data}")
    assert all_data == len(dic_judged_data)
    return dic_judged_data
        
def get_sm_notjudged_data(final_dict, judged_dict):
    remain2judge = {}
    judged_num, tojudged_num, all_data = 0, 0, 0
    for key in final_dict:
        if key in judged_dict.keys():
            # import pdb;pdb.set_trace()
            judged_num += 1
            all_data += 1
            continue
        # import pdb;pdb.set_trace()
        remain2judge[key]= final_dict[key]
        tojudged_num += 1
        all_data += 1
    
    assert tojudged_num + judged_num == len(final_dict) == all_data
    assert len(remain2judge) == tojudged_num
    print(f'还需要判断的个数为：{tojudged_num}, 已经判断的个数为:{judged_num}, 全部的个数为:{all_data}')


    return remain2judge


def data4llm(remain2judge, data, outfile):
    org_data_dic = {}
    for example in data:
        sen = example['sen']
        prd_word = example['prd_word']
        prd_idx = example['pred.idx']
        org_data_dic[(sen, prd_word, prd_idx)] = example
    print(f"原始句子中的个数为: {len(org_data_dic)}")


    new_core_2judge = []
    idx = 0
    # 转换成大模型可以判断的样子：
    for key in remain2judge:
        dic_core = {}
        idx += 1
        sen, prd_word, prd_idx, label, span_idx = key 
        ins = remain2judge[key]
        span = " ".join(sen.split()[span_idx[0]:span_idx[1]])
        dic_core["idx"] = idx
        dic_core['sen'] = sen
        dic_core['prd_word'] = prd_word
        if (sen, prd_word, prd_idx) in org_data_dic:
            temp_ins = org_data_dic[(sen, prd_word, prd_idx)]
        prd_lemma = temp_ins['prd_lemma']
        dic_core['prd_lemma'] = prd_lemma
        dic_core['prd_sense'] = temp_ins['prd.sense']
        dic_core['prd_idx']  = prd_idx
        dic_core['label'] = label
        dic_core['span'] = span
        dic_core['span_idx'] = [span_idx[0], span_idx[1]]
        dic_core['span_mean'] =  ins['span_mean']
        assert label in target_labels
        new_core_2judge.append(dic_core)
    assert len(new_core_2judge) == len(remain2judge)
    write_json(new_core_2judge, outfile)    

if __name__ == "__main__":
    # step 1: 
    domain = "tc"
    final_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/llm/tc/correct_data_{domain}_smnotrecall_filter.json")
    dic_final_data = get_corrected_dic(final_data)

    # step 2: 再获得所有大模型目前判断过的结果：
    gold_o1minierror = read_json(f"llm/incorrect_data_{domain}_gold.json")
    gold_o1miniright = read_json(f"llm/correct_data_{domain}_gold.json")
    sm_o1minierror = read_json(f"llm/incorrect_data_{domain}_smallmodel.json")
    sm_o1miniright = read_json(f"llm/correct_data_{domain}_smallmodel.json")
    sm_judged_data_dic = get_sm_judged_dic([gold_o1minierror, gold_o1miniright, sm_o1minierror, sm_o1miniright])

    need2judge = get_sm_notjudged_data(dic_final_data, sm_judged_data_dic)

    # step 3: 转换成大模型判断的格式：
    data = read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") # 用来找prd_lemma, prd_sense
    fileout = f"final_data/{domain}/test_{domain}_4llm_notjudged.conll"
    data4llm(need2judge,  data, fileout)

    # step 4: 在使用7.1llm_judge_smallmodel获得预测结果之后，我需要先解析数据，获得gold的结果然后加入到之前判断过的结果中，然后进行评估

    
    

