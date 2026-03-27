from config_data import read_json, read_pickle


def get_dic(data):
    dic_a = {}
    for ins in data:
        sen = ins['sen']
        prd_word = ins['prd_word']
        prd_idx = ins['prd_idx']
        label = ins['label']
        key = (sen, prd_word, prd_idx, label)
        dic_a[key]=ins
    return dic_a


a1  =  read_pickle(f"anno/tc/tc_annotation_smallmodel_removerepeate_322.pkl")
a2 = read_pickle(f"anno/tc/tc_annotation_single_gold_201.pkl")

a1_dic = get_dic(a1)
a2_dic = get_dic(a2)
count = 0
for key in a1_dic:
    if key in a2_dic:
        print(a1_dic[key])
        print(a2_dic[key])
        print("\n")
        count += 1

print(count)
