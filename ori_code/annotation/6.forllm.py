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
    noframe = 0
    core_num , all_num = 0, 0
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
        else:
            noframe += 1
    
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
            if label in roles:
                # if sen == "A note scribbled by an officer on the Russian nuclear submarine Kursk has revealed that at least 23 of its 118 crewmembers survived the explosions which sank the vessel ." and prd_word=="survived":
                #         import pdb;pdb.set_trace()
                dic_core['span_mean'] =  roles[label]
                
            else:
                dic_core['span_mean'] = None
            if label in core_label:
                new_data_core.append(dic_core)
                core_num += 1
            new_data_all.append(dic_core)
            all_num += 1
    
    write_json(new_data_core, outfile_core)
    write_json(new_data_all, outfile_all)
    print(f'No frame num in all predicate : {noframe}')
    print(f'Gold all predicate num :{all_num}, Gold core predicate num : {core_num}')



def get_small_model_index(data, frames, outfile_core, outfile_all, outfile_repeat):
    new_data_core = []
    new_data_all = []
    repeat_data, repeat_core, repeat_all = [], 0, 0
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    idx = 0
    semicrf_in_gold, semicrf_in_treecrf, treecrf_in_semicrf, treecrf_in_gold = 0, 0, 0, 0
    semicrf_num, treecrf_num = 0, 0
    semicrf_num_core, treecrf_num_core = 0, 0
    noframe, noframe_semicrf, noframe_treecrf = 0, 0, 0

    for key in data:
        sen = key['sen']
        prd_word = key['prd_word']
        prd_lemma = key['prd_lemma'] 
        prd_sense = key['prd.sense']
        prd_idx = key['pred.idx']
        gold = key['gold']
        semicrf = key['semicrf_label']
        treecrf = key['treecrf_label']
        roles = []
        if prd_lemma in frames:
            senses = frames[prd_lemma]
            
            for ins in senses:
                if ins['role_set_id'] == ".".join([prd_lemma, prd_sense]):
                    roles = ins['roles']
        else:
            noframe += 1
            
        
        for arg in semicrf:
            dic_core, dic_all = {}, {}
            idx += 1
            label = arg
            if label in gold and gold[label]==semicrf[label]:
                semicrf_in_gold += 1
            elif label  in treecrf and treecrf[label]==semicrf[label]:
                semicrf_in_treecrf += 1
            else:
                span = " ".join(sen.split()[semicrf[arg][0]: semicrf[arg][1]])
                dic_core["idx"] = idx
                dic_core['sen'] = sen
                dic_core['prd_word'] = prd_word
                dic_core['prd_lemma'] = prd_lemma
                dic_core['prd_sense'] = prd_sense
                dic_core['prd_idx']  = prd_idx
                dic_core['label'] = label
                dic_core['span'] = span
                dic_core['span_idx'] = [semicrf[arg][0], semicrf[arg][1]]
                if label in roles:
                    # import pdb;pdb.set_trace()
                    dic_core['span_mean'] =  roles[label]
                else:
                    dic_core['span_mean'] = None
                    noframe_semicrf += 1
                if label in core_label:
                    new_data_core.append(dic_core)
                    semicrf_num_core += 1
                new_data_all.append(dic_core)
                semicrf_num += 1
        for  arg in treecrf:
            dic_core, dic_all = {}, {}
            idx += 1
            label = arg
            if label in gold and gold[label]==treecrf[label]:
                treecrf_in_gold += 1
            elif label in semicrf and semicrf[label]==treecrf[label]:
                treecrf_in_semicrf += 1
            else:
                span = " ".join(sen.split()[treecrf[arg][0]: treecrf[arg][1]])
                dic_core["idx"] = idx
                dic_core['sen'] = sen
                dic_core['prd_word'] = prd_word
                dic_core['prd_lemma'] = prd_lemma
                dic_core['prd_sense'] = prd_sense
                dic_core['prd_idx']  = prd_idx
                dic_core['label'] = label
                dic_core['span'] = span
                dic_core['span_idx'] = [treecrf[arg][0], treecrf[arg][1]]
                if label in roles:
                    # import pdb;pdb.set_trace()
                    dic_core['span_mean'] =  roles[label]
                else:
                    dic_core['span_mean'] = None
                    noframe_treecrf += 1
                if label in core_label:
                    new_data_core.append(dic_core)
                    treecrf_num_core += 1
                new_data_all.append(dic_core)
                treecrf_num += 1


        for  arg in treecrf:
            dic_core, dic_all = {}, {}
            idx += 1
            label = arg
            if  label in semicrf and semicrf[label]==treecrf[label]  and (label in gold and gold[label]!=treecrf[label] or label not in gold):
                span = " ".join(sen.split()[treecrf[arg][0]: treecrf[arg][1]])
                dic_core["idx"] = idx
                dic_core['sen'] = sen
                dic_core['prd_word'] = prd_word
                dic_core['prd_lemma'] = prd_lemma
                dic_core['prd_sense'] = prd_sense
                dic_core['prd_idx']  = prd_idx
                dic_core['label'] = label
                dic_core['span'] = span
                dic_core['span_idx'] = [treecrf[arg][0], treecrf[arg][1]]
                if label in roles:
                    # import pdb;pdb.set_trace()
                    dic_core['span_mean'] =  roles[label]
                else:
                    dic_core['span_mean'] = None
                    noframe_treecrf += 1
                if label in core_label:
                    repeat_data.append(dic_core)
                    repeat_core += 1
                    new_data_core.append(dic_core)
                # repeat_data.append(dic_core)
                repeat_all += 1
                new_data_all.append(dic_core)
    
    write_json(new_data_core, outfile_core)
    write_json(new_data_all, outfile_all)    # 包含了核心和非核心
    write_json(repeat_data, outfile_repeat)# 下次需要和单独预测的放在一起
    print(f'No frame num in all predicate : {noframe}')
    print(f'Semicrf in gold :{semicrf_in_gold}, Semicrf in treecrf : {semicrf_in_treecrf}')
    print(f'Treecrf in gold :{treecrf_in_gold}, Treecrf in semicrf : {treecrf_in_semicrf}')
    print(f'Semicrf nonoverlap all num : {semicrf_num}, Treecrf nonoverlap all num : {treecrf_num}, all:{semicrf_num+treecrf_num}')
    print(f'Semicrf nonoverlap core num : {semicrf_num_core}, Treecrf nonoverlap core num : {treecrf_num_core}, all saperate core:{semicrf_num_core+treecrf_num_core}')
    print(f'不在gold中，但在semicrf和treecrf两者中的repeat 结果：{repeat_all}, 核心个数为：{repeat_core}')
    print(f'all core:{semicrf_num_core+treecrf_num_core + repeat_core}, all data: {semicrf_num + treecrf_num + repeat_all}')
    assert len(new_data_all)==(semicrf_num + treecrf_num + repeat_all)
    assert len(new_data_core) == (semicrf_num_core+treecrf_num_core + repeat_core)

def get_marginalprob_index(data, frames,file_out ):
    new_data_core = []
    new_data_all = []
    core_label = {'ARG0':0, 'ARG1':0, 'ARG2':0, 'ARG3':0, 'ARG4':0, 'ARG5':0}
    idx = 0
    noframe = 0
    core_num , all_num = 0, 0
    mp_in_gold, mp_in_semicrf, mp_in_treecrf = 0, 0, 0
    for key in data:
        sen = key['sen']
        prd_word = key['prd_word']
        prd_lemma = key['prd_lemma'] 
        prd_sense = key['prd.sense']
        prd_idx = key['pred.idx']
        gold = key['gold']
        semicrf = key['semicrf_label']
        treecrf = key['treecrf_label']
        high_mp = key['high_mp']
        
        roles = []
        if prd_lemma in frames:
            senses = frames[prd_lemma]
            for ins in senses:
                if ins['role_set_id'] == ".".join([prd_lemma, prd_sense]):
                    roles = ins['roles']
        else:
            noframe += 1

        for arg in high_mp:
            dic_core, dic_all = {}, {}
            idx += 1
            label = arg
            dic_core["idx"] = idx
            dic_core['sen'] = sen
            dic_core['prd_word'] = prd_word
            dic_core['prd_lemma'] = prd_lemma
            dic_core['prd_sense'] = prd_sense
            dic_core['prd_idx']  = prd_idx
            dic_core['label'] = arg
            if len(high_mp[arg])!=0:
                for temp in high_mp[arg]:
                    temp_idx = temp.split("\t")
                    span = " ".join(sen.split()[int(temp_idx[0]):int(temp_idx[1])])
                    dic_core['span'] = span
                    dic_core['span_idx'] = [int(temp_idx[0]),int(temp_idx[1])]
                    if label in roles:
                        dic_core['span_mean'] =  roles[label]
                    else:
                        dic_core['span_mean'] = None
                    if label in core_label:
                        core_num += 1
                        if label in gold and gold[label]==[int(temp_idx[0]),int(temp_idx[1])]:      
                            mp_in_gold += 1
                        elif label in semicrf and semicrf[label]==[int(temp_idx[0]),int(temp_idx[1])]:
                            mp_in_semicrf += 1
                            # import pdb;pdb.set_trace()
                        elif label in treecrf and treecrf[label]==[int(temp_idx[0]),int(temp_idx[1])]:
                            mp_in_treecrf += 1
                        else:
                            new_data_core.append(dic_core)
                            
                        new_data_all.append(dic_core)
                        all_num += 1
    
    write_json(new_data_core, file_out)
   
    print(f'No frame num in all predicate : {noframe}')
    print(f'Gold all predicate num :{all_num}, Gold core predicate num : {core_num}')
    print(mp_in_gold, mp_in_semicrf, mp_in_treecrf)
    print(core_num -mp_in_gold- mp_in_semicrf- mp_in_treecrf)
    print(all_num)



if __name__ == "__main__":
    domain = "tc" #"bn"
    
    frames = read_json("/data/ljwang/span-SRL-LLM/propbank_frames_main/frame_out/frames_info_3.4.json") 
    data = read_json(f"final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll")
    

    ## gold转换可以LLM处理的格式
    file_out =  f"final_data/{domain}/test_{domain}_4llm_core_gold.conll"
    file_outall = f"final_data/{domain}/test_{domain}_4llm_all_gold.conll"
    get_gold_index(data, frames,  file_out, file_outall)

    # 数据已保存到: final_data/test_bn_4llm_core_gold.conll
    # 数据已保存到: final_data/test_bn_4llm_all_gold.conll
    # No frame num in all predicate : 75
    # Gold all predicate num :4531, Gold core predicate num : 2132

    # 小模型
    file_out_smallmodel = f"final_data/{domain}/test_{domain}_4llm_core_smallmodel.conll"
    file_outall_smallmodel = f"final_data/{domain}/test_{domain}_4llm_all_smallmodel.conll"
    file_repeat = f"final_data/{domain}/test_{domain}_4llm_all_smallmodel_repeat.conll"
    get_small_model_index(data, frames, file_out_smallmodel, file_outall_smallmodel, file_repeat)


    # 数据已保存到: final_data/test_bn_4llm_core_smallmodel.conll
    # 数据已保存到: final_data/test_bn_4llm_all_smallmodel.conll
    # No frame num in all predicate : 75
    # Semicrf in gold :2494, Semicrf in treecrf : 343
    # Treecrf in gold :2550, Treecrf in semicrf : 343
    # Semicrf nonoverlap all num : 278, Treecrf nonoverlap all num : 251, all:529
    # Semicrf nonoverlap core num : 140, Treecrf nonoverlap core num : 129, all core:269


    # # # data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.5.conll")
    # data = read_json(f"final_data/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.1.conll")
    # file_out_prob = f"final_data/test_{domain}_4llm_core_high_prob.conll"
    # file_outall_prob = f"final_data/test_{domain}_4llm_all_high_prob.conll"
    # get_marginalprob_index(data, frames, file_out_prob)