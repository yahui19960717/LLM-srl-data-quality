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

def get_highprob_labels(distributions, data, index_data, path_out, threshold): # candidate是包含了所有gold结果、semicrf的预测结果等,找到预测结果高的
    new_data = []
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    num_core, num_overlap1, num_overlap2, num_overlap3, all_num = 0, 0, 0, 0, 0
    non_core_num =  0
    error_num = 0
    num_gt, num_lt = 0, 0 #所有大于某个值的结果
    num_gt_core, num_lt_core = 0, 0# 核心大于某个值的结果
    test_num = 0

    all_candidate_spans = distributions['all_candidate_spans']
    all_pred_spans = distributions['all_pred_spans']
    all_candidate_spans_dic = {temp_key:0 for temp_key in all_candidate_spans}
    print(all(temp_key in all_candidate_spans_dic for temp_key in all_pred_spans))
    all_candidate_spans_dic = {temp_key:0 for temp_key in all_candidate_spans}
    highprob_dic = defaultdict(lambda: defaultdict(dict))
    for key in all_candidate_spans_dic.keys():
        sen = " ".join(index_data[str(key[0])][0])
        prd = str(key[1])
        if key[-1] >= threshold:
            num_gt += 1
            highprob_dic[" ".join([sen, prd])][conll12_label[key[4]]]["\t".join([str(key[2]), str(key[3])])] = key[5]
            # highprob_dic_prob[" ".join([sen, prd])][conll12_label[key[4]]].append([key[2], key[3], key[5]])
            if conll12_label[key[4]] in core_label:
                num_gt_core += 1
        else:
            num_lt += 1
            if  conll12_label[key[4]] in core_label:
                num_lt_core += 1
  
    for key in data:
        sen = key['sen']
        prd = str(key['pred.idx'])
        temp = " ".join([sen, prd])
        if temp in highprob_dic.keys(): # 找到highprob对应的句子谓词
            key['high_mp'] = {}
            all_num += 1
            for ele in highprob_dic[temp]: # 遍历每个label
                # print(highprob_dic[temp]) # highprob过滤出来的spans，每个label可能对应不同的论元片段
                temp_high_mp  = highprob_dic[temp][ele].copy() #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                if ele in core_label.keys() :
                        if ele in key['gold'].keys() and "\t".join([str(x) for x in key['gold'][ele][:2]]) in highprob_dic[temp][ele]:
                            # gold里面存在这个标签且有
                            num_overlap1 += 1
                            temp_high_mp.pop("\t".join([str(x) for x in key['gold'][ele][:2]]))
                            key['high_mp'][ele]= temp_high_mp 
                            test_num +=   len(temp_high_mp )             
                        elif ele in key['semicrf_label'].keys() and "\t".join([str(x) for x in key['semicrf_label'][ele][:2]]) in highprob_dic[temp][ele]:
                            num_overlap2 += 1
                            temp_high_mp.pop("\t".join([str(x) for x in key['semicrf_label'][ele][:2]]))
                            key['high_mp'][ele]= temp_high_mp
                            test_num +=   len(temp_high_mp )  
                            # import pdb;pdb.set_trace()
                        elif ele in key['treecrf_label'].keys() and "\t".join([str(x) for x in key['treecrf_label'][ele][:2]]) in highprob_dic[temp][ele]:
                            num_overlap3 += 1
                            temp_high_mp.pop( "\t".join([str(x) for x in key['treecrf_label'][ele][:2]]))
                            key['high_mp'][ele]= temp_high_mp
                            test_num +=   len(temp_high_mp ) 
                        else:
                            key['high_mp'][ele]= temp_high_mp      # import pdb;pdb.set_trace()
                            test_num +=   len(temp_high_mp )  
                else:
                    # 被选择的500个句子中且在通过阈值筛选的候选 spans 的非核心spans的个数
                    non_core_num += len(highprob_dic[temp][ele])
        
        else: # 找不到highprob对应的句子谓词,被过滤掉了
            # print(temp)
            # print(key)
            error_num += 1
            key['high_mp'] = {}
        new_data.append(key)

    for key in new_data:
        for span_temp in key['high_mp']:
            num_core += len(key['high_mp'][span_temp])
            if len(key['high_mp'][span_temp]) != 0:
                print(key['sen'])
                print(key['gold'])
                print(key['semicrf_label'])
                print(key['treecrf_label'])
                print(key['high_mp'][span_temp])
                print("\n")

    write_json(new_data, path_out)
    assert len(data)  == error_num + all_num
    print(f'great than {threshold} : {num_gt}, less than {threshold}: {num_lt}')
    print(f'core span great than {threshold} : {num_gt_core}, core span less than {threshold}: {num_lt_core}')
    print(f"core : {num_core}, overlap : {num_overlap1, num_overlap2, num_overlap3}")
    print(f"被选择的500个句子中不在通过阈值筛选的候选 spans 的谓词数: {error_num}, 被选择的500个句子中在通过阈值筛选的候选 spans 的谓词数: {all_num}")
    print(f'被选择的500个句子中且在通过阈值筛选的候选 spans 的非核心spans的个数: {non_core_num}' )
    print(test_num)
    # assert temp_num == num_overlap1+num_overlap2+num_overlap3+num_core+non_core_num


            
# 看下是不是所有的预测结果都在候选结果中

if __name__=="__main__":
    domain = "tc" #"bn"
    distributions = read(f"../../prob_distribution/nw/nw-{domain}-test-distribution-maximum.pt") # 概率分布
    gold_semicrf_treecrf =  read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll")
    path_out = f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.1.conll"
    index_data = read_json(f"../../index_sen/nw/{domain}.test.json")
    get_highprob_labels(distributions, gold_semicrf_treecrf, index_data, path_out, threshold=0.1)