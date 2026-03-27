import json
import copy
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

def extract_args_all_preds(data_dict):
    """
    处理多个谓词，返回 {pred_idx: {role: [[start, end], ...]}}
    """
    result = {}
    
    for pred_idx, annotations in data_dict.items():
        pred_result = {}
        
        for token_idx, labels in annotations:
            for label in labels:
                if label == 'O' or not label:
                    continue
                
                if label.startswith('B-'):
                    role = label[2:]
                    if role not in pred_result:
                        pred_result[role] = []
                    pred_result[role] = [token_idx, token_idx + 1]
                    
                elif label.startswith('I-'):
                    role = label[2:]
                    if role in pred_result and pred_result[role]:
                        pred_result[role][1] = token_idx + 1
        
        result[pred_idx] = pred_result
    
    return result

def get_treecrf_results(gold_data, treecrf, path_out):
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    num_core, num_overlap = 0, 0
    new_data = []
    dic_treecrf = {}
    for element in treecrf:
        temp = extract_args_all_preds(element[-1]) 
        # {1: {'ARG1': [1, 4]}, 4: {'ARG0': [1, 2]}, 5: {'ARG1': [0, 4], 'ARGM-NEG': [5, 6], 'ARG2': [6, 9]}}
        sen = " ".join(element[0])
        for key in temp: 
            prd = key
            treecrf_lable = temp[key]
            dic_treecrf[" ".join([sen, str(prd)])] = treecrf_lable
    

    for key in gold_data:
        sen = key['sen']
        prd = key['pred.idx']
        temp = " ".join([sen, str(prd)])
        if temp in dic_treecrf.keys():
            key['treecrf_label'] = dic_treecrf[temp]
            new_data.append(key)
            for ele in dic_treecrf[temp].keys():
                if ele in core_label.keys() :
                    if ele in key['gold'].keys() and dic_treecrf[temp][ele] == key['gold'][ele]:
                        num_overlap += 1
                    elif ele in key['semicrf_label'].keys() and dic_treecrf[temp][ele] == key['semicrf_label'][ele]:
                        num_overlap += 1
                    else:
                        num_core += 1
            # import pdb;pdb.set_trace()
        else:
            raise KeyError(f"Key '{temp}' not found in treecrf_dic")

    
    write_json(new_data, path_out)
    print(f"core : {num_core}, overlap : {num_overlap}")
    # core : 129, overlap : 2042


if __name__=="__main__":
    domain = "tc" #bn"
    gold_treecrf = read_treecrf(f"/data/ljwang/span-SRL-LLM/data/{domain}/test_{domain}_bii.conll")
    pred_treecrf = read_treecrf(f"/data/ljwang/span-SRL-LLM/treecrf/pred_test/nw/nw_{domain}.conll")
    treecrf = get_prd_labels_dic(gold_treecrf, pred_treecrf)
    gold_semicrf =  read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel.conll")
    path_out = f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll"
    get_treecrf_results(gold_semicrf, treecrf, path_out)