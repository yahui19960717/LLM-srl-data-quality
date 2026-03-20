# 需要算一下整体的准确率就可以了，gold+我们新增的span
from config_data import read_json, write_json, read_pickle, build_gold_spans_from_dic, build_gold_dict_from_eval

def get_noin_gold_4llm(org_gold_data, current_gold_data, final_data):

    for key in current_gold_data:
        import pdb;pdb.set_trace()
        


if __name__=="__main__":
    domain = "bn"
    # 原始数据
    org_gold_data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll") 
    origin_gold_dict = build_gold_dict_from_eval(org_gold_data) 
    data = read_json( f"final_data/test_{domain}_4llm_core_gold.conll")
    
    # 最终标注的数据
    # eval_data = build_bigmodel_dict_from_eval_single(gold_bigmodel) # 仅gold结果
    final_data = read_pickle("annotated_final/final_annotated_all.pkl") # final annotated data bn
    current_gold_data = build_gold_spans_from_dic(final_data)
    get_noin_gold_4llm(org_gold_data, current_gold_data, final_data)


    







