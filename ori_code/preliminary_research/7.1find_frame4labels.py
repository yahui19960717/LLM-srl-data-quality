# 创建LLM需要的数据

import os
import json
import re
import spacy
from lemminflect import getLemma
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']

# 读conll文件
def read(file):
    with open(file, "r", encoding="utf-8") as f:
        sentences = []
        word, prd, srl, index_prd = [], [], [], []
        i = 0
        for line in f:
            temp = line.strip().split()
            if len(temp)==0:
                arg = [list(i) for i in zip(*srl)]
                sentences.append((word, prd, index_prd, arg))
                word, prd, srl, index_prd = [], [], [], []
                i = 0 
            else:
                word.append(temp[1])
                if temp[2]!= "-":
                    prd.append(temp[2])
                    index_prd.append(i)
                srl.append(temp[4:])
                i += 1
    # print(len(sentences))
    return sentences
# 写json文件 
def write_json(data, file):
    with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=0)
# 读json文件 index
def read_json(file):
  with open(file, "r", encoding="utf-8") as f:
      data = json.load(f)
  return data 

# 1. 缩写还原函数
def normalize_contractions(text: str) -> str:
    # 把 “I 'm” 中间的空格去掉
    text = re.sub(r"\b(\w+)\s+'(\w+)\b", r"\1'\2", text)

    # 2. 把口头填充词 % um / %uh / % er 等归一化：%um -> um
    text = re.sub(r'%\s*(um|uh|er|ah|mm|eh)', r'\1', text, flags=re.I)

    # 3. 定义“一个 token”的正则：以 * 开头/结尾，中间只有字母
    star_word = re.compile(r'^\*[a-zA-Z]+\*$')
    # 如果有其他缩写，可以继续加规则
    return text

def get_lemma(sentences, frames, nlp): ## 获得每个谓词的lemma形式，并判断是否有对应的frame
    count, org_count, change_count, not_find_count=0, 0, 0, 0
    new_data = []
    
    ## 获得每个谓词的lemma形式
    for s in sentences:
        # import pdb;pdb.setc_trace()
        pre = s['predicate'].lower()
        sen = s['sentences']
        pred_id=s['org_span'][1]-1 # 这个是原始的spans

        # 处理一些谓词是缩写的情况
        if pre == "'s":
            pre = "is"
        if pre == "'ve":
            pre = "have"

        if pre in frames.keys():
            org_count += 1
            count += 1
            s['lemma']=pre
        else:
            # 处理句子中特殊符号的情况，spacy一般将其作为单独的字来处理
            sen = normalize_contractions(sen)
            sen_pos = nlp(sen)
            token = sen_pos[pred_id]._.lemma() # lemma

            if token in frames.keys():
                change_count += 1
                count += 1
            else: # 对于有问题的句子，可以直接让其谓词直接转换成lemma，使用的是内置lemma
                sen_pos = nlp(pre)
                token = sen_pos[0].lemma_
                if token in frames.keys():
                    change_count += 1
                    count += 1
                else:
                    not_find_count += 1
                    print(pre, token)
                    print("not find")  
                    
            s['lemma']=token
        new_data.append(s)
    assert len(sentences) == len(new_data)
    if not_find_count==0:
        print(f"数据中的谓词都可以在找到frame！！")
    print(f'句子个数：{len(sentences)}, 谓词个数：{count}, \n谓词原型个数：{org_count}, \n谓词改变的个数：{change_count}, \n没有找到的个数：{not_find_count}')
    return new_data

def get_frame(sentences, orginal_sen, frames, file=None): # 获得对应谓词sense的id及example

    # 先获得原始句子的谓词 sense id
    dic_sen = {}
    org_sen = read(orginal_sen)
    for s in org_sen:
        sen = " ".join(s[0])
        preds = s[2]
        for i, p in enumerate(preds):
            pred_sen = s[1][i]
            key = "\t".join([sen, str(p)]) # 句子+谓词index
            dic_sen[key] = pred_sen
    
    print(f"数据集中谓词的个数为：{len(dic_sen)}")

    
    # 遍历选择的span，来查找对应的sense,并添加sense
    new_sentences = []
    for s in sentences:
        sen = s['sentences']
        p = s['org_span'][1]
        key = "\t".join([sen, str(p-1)]) 
        if key in dic_sen.keys():
            s['pred_sense'] = dic_sen[key]
        else:
            print("警告：⚠️句子有问题，在test文件中没有找到它！")
        
        new_sentences.append(s)
    assert len(new_sentences) == len(sentences)

    # 根据找到的sense来找frame中对应的example、以及core argument role labels
    count = 0
    final_results = []
    for s in new_sentences:
        if s['lemma'] in frames.keys():
            pred_sense = s['pred_sense'] 
            sense = ".".join([s['lemma'], str(pred_sense)])
            flag = 0
            for temp in frames[s['lemma']]:
                if sense == temp['role_set_id']:
                    flag = 1
                    candidate_roles = temp['roles']
                    examples = temp['examples']
                    s['candidate_roles']= candidate_roles
                    s['examples']=examples
            if flag != 1: # 说明找不到lemma的sense
                print('*'*15)
                print(f"句子:{s['sentences']}\n谓词:{s['predicate']}({s['lemma']})\n谓词含义id:{s['pred_sense']}/{sense}找不到!")
                print('*'*15)
                count += 1
        else:
            print('!'*15)
            print(f"句子:{s['sentences']}\n谓词:{s['predicate']}({s['lemma']})\n谓词含义id:{s['pred_sense']}找不到！")
            print('!'*15)
            count+=1
        final_results.append(s)
    print(f"没有对应谓词词义的个数为: {count}")


    write_json(final_results, file)




    # 选择出来的结果来跑的
    

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 只需要在test上做实验吧？ 
    source = ["nw", "bn", "bc"]# 
    target = ['tc', 'bn', 'nw', 'bc'] #
    nlp = spacy.load('en_core_web_sm')
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                file_save = f'../forllm_unite_newest/{i}/{i}-{j}-{k}.json'
                sentences = read_json(file_save) # sentences info: include sen, prd, span, error_type and so on;span-level
                frames = read_json("../propbank-frames-main/frame_out/frames_info_3.4.json") # frame info
                new_sentences = get_lemma(sentences, frames, nlp)
                # 获得id的句子
                data_id = f"/data/yhliu/Span-based-SRL/data/{j}/{k}_{j}.conll"
                path_write = f'../forllm_frames_newest/{i}/{i}-{j}-{k}.json'
                get_frame(new_sentences, data_id, frames, path_write)
            

