"""
解析 bn.conll 文件：
- 每个句子由空行分隔
- 每行 token：col0=索引, col1=单词, col2=义项(或"-"), col3=词元(或"-"), col4+= 各谓词标注列
- 谓词行：col2 != "-" 的行，按出现顺序依次对应 col4, col5, col6 ...
- 标注格式：括号嵌套的 span 标签，如 (ARG0*), (V*), (ARG1* ... *)
span的下标从0开始，结束就是最后的词，但是注意semicrf等都是结束词的下一个下标。谓词的下标从1开始
"""

from dataclasses import dataclass 
from typing import Optional
import json

@dataclass
class Token:
    index: int        # 句内 token 下标 (1-based)
    word: str
    sense: Optional[str]   # 义项编号，None 表示非谓词
    lemma: Optional[str]   # 词元


@dataclass
class Predicate:
    token_index: int       # 谓词在句中的 token 下标 (1-based)
    lemma: str
    sense: str
    word: str
    arguments: dict        # {角色标签: [token_index, ...]}  e.g. {"ARG0": [1,2], "V": [6]}


@dataclass
class Sentence:
    tokens: list           # List[Token]
    predicates: list       # List[Predicate]

    def text(self):
        return " ".join(t.word for t in self.tokens)


def parse_span_column(labels: list[str]) -> dict[str, list[int]]:
    """
    解析单个谓词标注列，返回 {角色: [token_index, ...]}。
    标注格式示例：
      (ARG0*)  -> 单 token span
      (ARG1*   -> span 开始-1
      *)       -> span 结束-1
      *        -> 不属于任何 span（或在已开启的 span 内部）
    """
    args = {}
    stack = []  # 当前打开的 span 标签

    for idx, label in labels:
        label = label.strip()
        # 一个 cell 可能同时开启和关闭，如 (ARG0*)
        # 也可能只开启 (ARG1*  或只关闭 *)

        # 统计开启的标签
        opens = []
        rest = label
        while rest.startswith("("):
            star_pos = rest.find("*")
            tag = rest[1:star_pos]  # 如 "ARG0", "V", "ARGM-TMP"
            opens.append(tag)
            rest = rest[star_pos:]  # 剩余从 * 开始

        # 把新开启的 span 压栈
        for tag in opens:
            stack.append((tag, idx-1))
            if tag not in args:
                args[tag] = []

        # # 当前 token 属于栈中所有打开的 span
        # for tag in stack:
        #     args[tag].append(idx-1)

        # 统计关闭次数
        close_count = rest.count(")")
        for _ in range(close_count):
            if stack:
                tag, start = stack.pop()
                args[tag] =   [start, idx]
    return args


def parse_conll(filepath: str) -> list[Sentence]:
    sentences = []
    current_lines = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current_lines:
                    sentences.append(_build_sentence(current_lines))
                    current_lines = []
            else:
                current_lines.append(line)
    if current_lines:
        sentences.append(_build_sentence(current_lines))
    print(f"句子个数为：{len(sentences)}")
    return sentences


def _build_sentence(lines: list[str]) -> Sentence:
    tokens = []
    # 先确定哪些行是谓词行（col2 != "-"），按顺序对应标注列
    predicate_rows = []  # (行号, Token)

    for line in lines:
        fields = line.split("\t")
        idx = int(fields[0])
        word = fields[1]
        sense = fields[2] if fields[2] != "-" else None
        lemma = fields[3] if fields[3] != "-" else None
        tok = Token(index=idx, word=word, sense=sense, lemma=lemma)
        tokens.append(tok)
        if sense is not None:
            predicate_rows.append(tok)

    num_predicates = len(predicate_rows)

    # 收集每个谓词标注列的原始标签
    # 标注列从 col4 开始，第 i 个谓词对应 col(4+i)
    pred_columns: list[list[tuple[int, str]]] = [[] for _ in range(num_predicates)]

    for line in lines:
        fields = line.split("\t")
        idx = int(fields[0])
        for pi in range(num_predicates):
            col = 4 + pi
            if col < len(fields):
                pred_columns[pi].append((idx, fields[col]))

    # 构建 Predicate 对象
    predicates = []
    for pi, ptok in enumerate(predicate_rows):
        args = parse_span_column(pred_columns[pi])
        predicates.append(Predicate(
            token_index=ptok.index,
            lemma=ptok.lemma,
            sense=ptok.sense,
            word=ptok.word,
            arguments=args,
        ))

    return Sentence(tokens=tokens, predicates=predicates)



def print_predicate(sent: Sentence, pred: Predicate):
    """打印一个谓词及其标注结果。"""
    print(f"  谓词: {pred.word} (lemma={pred.lemma}, sense={pred.sense}, token_index={pred.token_index})")
    for role, indices in pred.arguments.items():
        span_words = [sent.tokens[i - 1].word for i in indices]
        print(f"    {role}: {' '.join(span_words)}  (tokens: {indices})")


# ---- 演示 ----
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")
if __name__ == "__main__":
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    domain = 'tc'
    number_core = 0
    filepath = f"selected_data/test_{domain}_selected_data_692.conll"
    fileout = f"final_data/test_{domain}_goldlabel.conll"
    sentences = parse_conll(filepath)
    prd_all_num, prd_rm_be_num = 0, 0
    '''  具体说明：
        - sent.text() — 返回拼接好的句子字符串，如 "Tomorrow 's summit meeting will bring Ehud Barak and Yasser Arafat to the resort city of Sharm El - eikh ."
        - sent.predicates — 句中所有谓词的列表，每个 Predicate 包含 word、lemma、sense、token_index
        - pred.arguments — 该谓词的标注结果，是一个字典，key 是角色标签（如 "ARG0", "V", "ARG1"），value 是对应的 token 下标列表（1-based），例如 {"ARG0": [1,2,3,4],
        "V": [6], "ARG1": [7,8,9,10,11]}
    '''
    temp_list = []
    for sent in sentences:                                                                                                                                     
        # 1) 句子字符串
        print(sent.text())

        # 2) 每个谓词 & 3) 该谓词对应的标注序列
        for pred in sent.predicates:
            print(pred.word, pred.lemma, pred.sense, pred.token_index)  # 谓词本身
            dic_temp = {}
            prd_all_num += 1
            if pred.lemma != "be":
                prd_rm_be_num += 1
                dic_temp['sen'] = sent.text()
                dic_temp['prd_word']  = pred.word
                dic_temp['prd_lemma'] = pred.lemma
                dic_temp['prd.sense']  = pred.sense
                dic_temp['pred.idx'] = pred.token_index
                dic_temp['gold'] = pred.arguments
                print(pred.arguments) 
                for key in pred.arguments:
                    if key in core_label.keys():
                        number_core += 1
                    # import pdb;pdb.set_trace() 
                temp_list.append(dic_temp)
             
                               # 标注: {角色: [token下标列表]}

    write_json(temp_list, fileout)
    print(f"共解析 {len(sentences)} 个句子，{sum(len(s.predicates) for s in sentences)} 个谓词\n")
    print(f"去除 be 动词之后的谓词个数为 ： {prd_rm_be_num}")
    print(f'全部的谓词个数为：  {prd_all_num}')
    print(f'gold core number is : {number_core}')
    # bn
    # 去除 be 动词之后的谓词个数为 ： 1455
    # 全部的谓词个数为：  1766    
    # 2132

  