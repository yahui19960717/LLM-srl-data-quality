from typing import Any


import json
import random
random.seed(1)  #  
#   - Column 1: token index
#   - Column 2: word                                                                                                                                             
#   - Column 3: sense number (or -)                                                                                                                            
#   - Column 4: lemma (or -)
#   - Predicates are identified by having a non-- value in column 3 (the sense column), and column 4 gives the lemma.

def parse_conll(filepath):
    """解析 CoNLL 文件，返回句子列表。每个句子是 token 行的列表。"""
    sentences = []
    current = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(line)
    if current:
        sentences.append(current)
    return sentences


def get_predicates(sentence):
    """提取句子中所有谓词的 lemma。谓词行的第3列(sense)不为 '-'。"""
    predicates = []
    for line in sentence:
        fields = line.split("\t")
        if len(fields) >= 4 and fields[2] != "-":     
            predicates.append(fields[3])  # lemma
    return predicates


def has_non_be_predicate(sentence):
    """判断句子是否包含至少一个非 be 动词的谓词。"""
    preds = get_predicates(sentence)
    if not preds: # 去掉了没有谓词的句子和谓词全部是be动词的句子
        return False
    return any(p != "be" for p in preds)


def sentence_text(sentence):
    """将句子还原为文本。"""
    words = []
    for line in sentence:
        fields = line.split("\t")
        if len(fields) >= 2:
            words.append(fields[1])
    return " ".join(words)

def write_text(sentences,file):
    with open(file, "w", encoding="utf-8") as f:
        for s in sentences:
            for line in s:
                f.write(line)
                f.write("\n")
            f.write('\n')
def main(path, num_data):
    filepath = f"/data/ljwang/span-SRL-LLM/data/{path}/test_{path}.conll"
    file_out = f"selected_data/test_{path}_selected_data_{num_data}.conll"
    print("保存路径：", f"selected_data/test_{path}_selected_data_{num_data}.conll")
    sentences = parse_conll(filepath) # 每个句子是n行
    print(f'{filepath}-> 句子的个数为：{len(sentences)}')

    # 筛选：不能所有谓词都是 be. 
    candidates = [s for s in sentences if has_non_be_predicate(s)] #  

    if len(candidates) < num_data:
        print(f"符合条件的句子不足{num_data}个。只有{len(candidates)}")
        return

    sampled = random.sample(candidates, num_data)

    print(len(sampled))
    write_text(sampled, file_out)
    exit()
    for i, sent in enumerate[Any](sampled, 1):
        preds = get_predicates(sent)
        print(f"--- 句子 {i} ---")
        print(f"文本: {sentence_text(sent)}")
        print(f"谓词: {preds}")
        print()


if __name__ == "__main__":
    # main("bn", 500)
    main("tc", 692)
