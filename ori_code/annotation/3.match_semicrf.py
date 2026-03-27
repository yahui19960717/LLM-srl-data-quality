import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

def get_semicrf_results(gold_data, index_data, path):
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    num_core, num_overlap = 0, 0
    semicrf_dic  = {}
    new_data = []
    for key in index_data.keys():
        sen = " ".join(index_data[key][0])
        prd = index_data[key][1][0][0]
        temp = "\t".join([sen, str(prd)])
        semicrf_dic[temp] = index_data[key]
    
    for key in gold_data:
        sen = key['sen']
        prd_idx = key['pred.idx']
        temp = "\t".join([sen, str(prd_idx)])
        if temp in semicrf_dic.keys():
            argument = semicrf_dic[temp][1]
            args_dic = {}
            for arg in argument:
                if arg[3] != "O":
                    # import pdb;pdb.set_trace()
                    args_dic[arg[3]] = [arg[1], arg[2]]
            key['semicrf_label'] = args_dic
            for ele in args_dic.keys():
                if ele in core_label.keys() :
                    if ele in key['gold'].keys() and args_dic[ele] == key['gold'][ele]:
                        num_overlap += 1
                    else:
                        num_core += 1

            new_data.append(key)
            # # import pdb;pdb.set_trace()
            # print(key['gold'])
            # print(semicrf_dic[temp][3])
            # import pdb;pdb.set_trace()
        else:
            raise KeyError(f"Key '{temp}' not found in semicrf_dic")

    write_json(new_data, path)
    print(f"core : {num_core}, overlap : {num_overlap}")


if __name__=="__main__":
    domain = "tc" #bn"
    index_data = read_json(f"../../index_sen/nw/{domain}.test.json")
    gold_data = read_json(f"final_data/test_{domain}_goldlabel.conll")
    path_out = f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel.conll"
    get_semicrf_results(gold_data, index_data, path_out)
    # core : 319, overlap : 1806
    #2125
      