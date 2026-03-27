import os
import json
import pickle
from config_data import read_pickle,read_json
'''
统计下bn最终的数据
'''

def statistic_final(data):
    print(f'所有标注个数为:{len(data)}')
    error_data, mark_optional = 0, 0
    multi_span_num, no_span_num, one_span_num = 0, 0, 0
    for key in data:
        sen, word, prd_idx, label = key.split("\t")
        ins = data[key]
        if ins["grammar_status"] == "有语法错误": # 有语法错误的句子
            error_data += 1
        else:
            if ins.get('optional',None) == True:
                mark_optional += 1
            else:
                if len(ins['selected_spans']) ==0:
                    no_span_num+=1
                elif len(ins['selected_spans']) >=2:
                    multi_span_num+=1
                elif len(ins['selected_spans']) == 1:
                    one_span_num += 1
                else:
                    import pdb;pdb.set_trace()



    print(f"有语法错误的句子个数：{error_data}")
    print(f'可标可不标的句子个数：{mark_optional}')
    print(f'多个候选span的个数：{multi_span_num}, 单个span的个数：{one_span_num}, 无论元的个数:{no_span_num}')


if __name__ == "__main__":
    domain = "bn"
    corrected_data =  read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/analysis/test_{domain}_500_core_final_v4.json")
    statistic_final(corrected_data)
   
