import json 
import torch
import copy
from collections import defaultdict
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
# 获得marginal probability大于0.8的结果
def read(file,device="cuda"):
    data = torch.load(file,map_location=device,mmap=True, weights_only=True)
    return data

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

def read_treecrf(file):
    prd_num = 0
    with open(file, "r", encoding="utf-8") as f:
        sentences, words, srls, prds, prd_idx = [], [], [], [], {}
        i = 0
        for line in f:
            temp = line.strip().split()
            # [id, words, words, _*5, srl, _]
            if len(temp)==0:
                sentences.append([words, srls, prds, prd_idx])
                words, srls, prds, prd_idx =  [], [], [], {}
                i = 0
            else:
                i += 1
                words.append(temp[1])
                srls.append(temp[8])
                if "0:[prd]" in temp[8]:
                    prd_idx[i] = []
                    prds.append(temp[1])
                    prd_num += 1
    print(f'谓词个数为：{prd_num}')
    return sentences

def get_prd_labels_dic(gold_sentences, pred_sentences):
    # 用来获得srl 指向了同样的对象
    for i_gold, sen in enumerate(gold_sentences):
        # temp_srl = sen[1]
        temp_srl = pred_sentences[i_gold][1]
        prd_idx = copy.deepcopy(sen[3]) # 防止指向同一个对象
        # 每个谓词的标注已获得
        for idx, ele in enumerate(temp_srl):
            if ele != "_":
                temp_srl = ele.split("|")
                for label in temp_srl:
                    prd_label = (idx, label.split(":")[1:])
                    try:
                        if int(label.split(":")[0]) != 0: # 这里有问题！！！！
                            prd_idx[int(label.split(":")[0])].append(prd_label)
                    except:
                        continue
                        print("ERROR")
        pred_sentences[i_gold][3] = prd_idx
    return pred_sentences

def get_highprob_labels(distributions, data, index_data, path_out): # candidate是包含了所有gold结果、semicrf的预测结果等,找到预测结果高的
    new_data = {}
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    num_core, num_overlap1, num_overlap2, num_overlap3, all_num = 0, 0, 0, 0, 0
    non_core_num =  0
    error_num = 0
    num_gt, num_lt = 0, 0
    core_num_highprob = 0
    data_in_highprob_arg_num = 0
    temp_num = 0
    all_candidate_spans = distributions['all_candidate_spans']
    all_pred_spans = distributions['all_pred_spans']
    all_candidate_spans_dic = {temp_key:0 for temp_key in all_candidate_spans}
    print(all(temp_key in all_candidate_spans_dic for temp_key in all_pred_spans))
    all_candidate_spans_dic = {temp_key:0 for temp_key in all_candidate_spans}
    highprob_dic = defaultdict(dict)
    for key in all_candidate_spans_dic.keys():
        sen = " ".join(index_data[str(key[0])][0])
        prd = str(key[1])
        if key[-1] >= 0.5:
            num_gt += 1
            highprob_dic[" ".join([sen, prd])][conll12_label[key[4]]] = [key[2], key[3]]
            if conll12_label[key[4]] in core_label:
                core_num_highprob += 1
        else:
            num_lt += 1
            highprob_dic[" ".join([sen, prd])] = {}
        # if len(highprob_dic[" ".join([sen, prd])]) >= 2:
        #     import pdb;pdb.set_trace()
    for key in data:
        sen = key['sen']
        prd = str(key['pred.idx'])
        temp = " ".join([sen, prd])
        if temp in highprob_dic.keys():
            key['high_mp'] = {}
            all_num += 1
            data_in_highprob_arg_num += len(highprob_dic[temp])
            for ele in highprob_dic[temp]:
                temp_num += 1
                # import pdb;pdb.set_trace()
                if ele in core_label.keys() :
                        if ele in key['gold'].keys() and highprob_dic[temp][ele] == key['gold'][ele]:
                            num_overlap1 += 1
                        elif ele in key['semicrf_label'].keys() and highprob_dic[temp][ele] == key['semicrf_label'][ele]:
                            num_overlap2 += 1
                        elif ele in key['treecrf_label'].keys() and highprob_dic[temp][ele] == key['treecrf_label'][ele]:
                            # print(3)
                            num_overlap3 += 1
                            # import pdb;pdb.set_trace()
                        else:
                            num_core += 1 # 不重叠的
                            key['high_mp'][ele]= highprob_dic[temp][ele]
                            # import pdb;pdb.set_trace()
                else:
                    # 对核心论元都是比较确性的
                    non_core_num += 1
        else:   
            print(temp)
            print(key)
            error_num += 1

            # raise KeyError(f"Key '{temp}' not found in highprob_dic")

    assert len(data)  == error_num + all_num
    write_json(new_data, path_out)
    print(f"core : {num_core}, overlap : {num_overlap1, num_overlap2, num_overlap3}, all num: {all_num}")
    print(f"error num : {error_num}")
    print(f'non core num : {non_core_num}' )
    print(f'great than 0.5 : {num_gt}, less than 0.5: {num_lt}')
    print(core_num_highprob) #  
    print(f' data_in_highprob_arg_num : {data_in_highprob_arg_num}')
    print(temp_num)
    assert temp_num == num_overlap1+num_overlap2+num_overlap3+num_core+non_core_num

# core : 0, overlap : 105, all num: 1438
# error num : 17
# non core num : 31
# great than 0.5 : 7742, less than 0.5: 1000748
            
# 看下是不是所有的预测结果都在候选结果中

if __name__=="__main__":
    domain = "bn"
    distributions = read(f"../../prob_distribution/nw/nw-{domain}-test-distribution-maximum.pt") # 概率分布
    gold_semicrf_treecrf =  read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll")
    path_out = f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob.conll"
    index_data = read_json(f"../../index_sen/nw/{domain}.test.json")
    get_highprob_labels(distributions, gold_semicrf_treecrf, index_data, path_out)