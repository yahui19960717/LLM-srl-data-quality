import json
import random 
random.seed(1)

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")


def get_gold_index(data, frames, outfile_core, outfile_all):
    new_data_core = []
    new_data_all = []
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    idx = 0
    for key in data:
        sen = key['sen']
        prd_word = key['prd_word']
        prd_lemma = key['prd_lemma'] 
        prd_sense = key['prd.sense']
        prd_idx = key['pred.idx']
        gold = key['gold']
        roles = []
        if prd_lemma in frames:
            senses = frames[prd_lemma]
            for ins in senses:
                if ins['role_set_id'] == ".".join([prd_lemma, prd_sense]):
                    roles = ins['roles']
            
        
        for arg in gold:
            dic_core, dic_all = {}, {}
            idx += 1
            label = arg
            span = " ".join(sen.split()[gold[arg][0]: gold[arg][1]])
            dic_core["idx"] = idx
            dic_core['sen'] = sen
            dic_core['prd_word'] = prd_word
            dic_core['prd_lemma'] = prd_lemma
            dic_core['prd_sense'] = prd_sense
            dic_core['prd_idx']  = prd_idx
            dic_core['label'] = label
            dic_core['span'] = span
            dic_core['span_idx'] = [gold[arg][0], gold[arg][1]]
            if label in ins['roles']:
                # import pdb;pdb.set_trace()
                dic_core['span_mean'] =  ins['roles'][label]
            else:
                dic_core['span_mean'] = None
            if label in core_label:
                new_data_core.append(dic_core)
            new_data_all.append(dic_core)
    
    write_json(new_data_core, outfile_core)
    write_json(new_data_all, outfile_all)
            


if __name__ == "__main__":
    domain = "bn"
    data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll")
    frames = read_json("/data/ljwang/span-SRL-LLM/propbank_frames_main/frame_out/frames_info_3.4.json") 
    file_out = path_out = f"final_data/test_{domain}_4llm_core_gold.conll"
    file_outall = f"final_data/test_{domain}_4llm_all_gold.conll"
    get_gold_index(data, frames,  file_out, file_outall)