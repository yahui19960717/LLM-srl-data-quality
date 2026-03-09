'''
需要将以下数据进行标注：
1）o1mini认为gold不正确的且deepseek认为不正确的（减少人工标注）
2) o1mini认为预测结果正确的（两个模型认为正确）
3）o1mini认为预测结果不正确，但deepseek认为正确的（两个模型都认为正确，为了召回）
若有两个及三个不同的结果就显示出来，否则就不给出选项
根据sen排序,将相同的sen放在一起
gold没有指的是label_span整体没有，但它可能有对应的label
'''
import json
from collections import defaultdict  
import random
import pickle
random.seed(1)

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_pickle(sentences, path):
    with open(path, 'wb') as f:
        pickle.dump(sentences, f)


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
    print(type)
    return new_data, choice_multi, choice_1



def deal_data(i1aidgold, all_data, fileout):
    print(f"o1mini和deepseek都认为gold结果不正确：{len(i1aidgold)} ")
    dic_alldata = {"\t".join([i['sen'], str(i['pred.idx'])]):i for i in all_data}
    new_data = []
    idx = 0
    choice_multi, choice_1 = 0, 0

    # sm认为正确，o1mini认为正确(deeseep可能认为正确也可能认为不正确，gold里面没有的)
    # new_data, choice_multi, choice_1  = get_annotation_instance(correct_sm_o1mini, new_data,dic_alldata,choice_multi, choice_1, type="sm_right, o1mini_right(deepseek)")
    # choice_multi_temp1, choice_1_temp_1 = choice_multi, choice_1
    # print(f'{len(new_data)}, {choice_multi_temp1}, {choice_1_temp_1}')
    # assert len(new_data)== len(correct_sm_o1mini)

    # # sm认为正确, deepseek认为正确，但o1mini认为不正确 （gold里面没有的span） 
    # new_data, choice_multi, choice_1 = get_annotation_instance(ibutdcsm, new_data,dic_alldata,choice_multi, choice_1, type="sm_right, o1mini_wrong，deepseek_right")   
    # choice_multi_temp2, choice_1_temp_2 = choice_multi, choice_1
    # print(len(new_data), choice_multi_temp2-choice_multi_temp1,  choice_1_temp_2-choice_1_temp_1)
    # assert len(new_data)== len(correct_sm_o1mini)+len(ibutdcsm)

    # gold正确，o1mini认为不正确，deepseek认为不正确
    new_data, choice_multi, choice_1 = get_annotation_instance(i1aidgold, new_data,dic_alldata,choice_multi, choice_1, type="gold_right, o1mini_wrong，deepseek_wrong") 
    choice_multi_temp3, choice_1_temp_3 = choice_multi, choice_1
    print(len(new_data), choice_multi_temp3-choice_multi,  choice_1_temp_3-choice_1) 
    assert len(new_data)== len(i1aidgold)
    print(len(new_data), choice_multi, choice_1)

    # 根据sen排序,将相同的sen放在一起
    sorted_data = sorted(new_data, key=lambda x: (x['sen'], str(x['prd_idx'])))

    newest_data = []
    idx=0

    temp_dic = {}
    for temp in sorted_data:
        sen = temp['sen']
        prd = str(temp['prd_idx'])
        label = temp['label']
        temp_key = "\t".join([sen, prd, label])
        temp_dic[temp_key]  = temp
    
    print(f' all argument number: {len(sorted_data)}, remove repeat: {len(temp_dic)}')
    # for instance in sorted_data:
    #     idx += 1
    #     new_instance= {'idx':idx} | instance
    #     newest_data.append(new_instance)

    for instance in temp_dic:
        idx += 1
        new_instance= {'idx':idx} | temp_dic[instance]
        newest_data.append(new_instance)


    
    write_pickle(newest_data, fileout)




def deal_random_data(i1aidgold, all_data, fileout):
    print(f"o1mini认为gold结果不对，但deepseek认为gold结果正确：{len(i1aidgold)} ")
    dic_alldata = {"\t".join([i['sen'], str(i['pred.idx'])]):i for i in all_data}
    new_data = []
    idx = 0
    choice_multi, choice_1 = 0, 0

    # gold正确，o1mini认为不正确，deepseek认为正确
    new_data, choice_multi, choice_1 = get_annotation_instance(i1aidgold, new_data,dic_alldata,choice_multi, choice_1, type="gold_right, o1mini_wrong，deepseek_right") 
    choice_multi_temp3, choice_1_temp_3 = choice_multi, choice_1
    print(len(new_data), choice_multi_temp3-choice_multi,  choice_1_temp_3-choice_1) 
    assert len(new_data)== len(i1aidgold)
    print(len(new_data), choice_multi, choice_1)
    random_new_data = random.sample(new_data, 30)

    # 根据sen排序,将相同的sen放在一起
    sorted_data = sorted(random_new_data, key=lambda x: (x['sen'], str(x['prd_idx'])))

    newest_data = []
    idx=0

    temp_dic = {}
    for temp in sorted_data:
        sen = temp['sen']
        prd = str(temp['prd_idx'])
        label = temp['label']
        temp_key = "\t".join([sen, prd, label])
        temp_dic[temp_key]  = temp
    
    print(f' all argument number: {len(sorted_data)}, remove repeat: {len(temp_dic)}')
    # for instance in sorted_data:
    #     idx += 1
    #     new_instance= {'idx':idx} | instance
    #     newest_data.append(new_instance)

    for instance in temp_dic:
        idx += 1
        new_instance= {'idx':idx} | temp_dic[instance]
        newest_data.append(new_instance)


    
    write_pickle(newest_data, fileout)



if __name__=="__main__":
    # domain = "bn"
    # incorrecto1mini_and_deepseekincrrect_gold = read_json(f"llm/incorrect_data_{domain}_gold_deepseek.json")
    # all_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.8.conll")
    # fileout = f"anno/bn_annotation_single_gold.pkl"
    # deal_data(incorrecto1mini_and_deepseekincrrect_gold, all_data, fileout)



    # o1mini认为不对，但deepseek认为正确的，随机选择30个句子来判断
    domain = "bn"
    incorrecto1mini_and_deepseekincrrect_gold = read_json(f"llm/correct_data_{domain}_gold_deepseek.json")
    all_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.8.conll")
    fileout = f"anno/bn_annotation_single_gold_random.pkl"
    deal_random_data(incorrecto1mini_and_deepseekincrrect_gold, all_data, fileout)
