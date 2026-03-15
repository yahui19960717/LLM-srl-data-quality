'''
对于Golden结果，重点看修复后的结果，与原始结果对比，如果修复后的有任何一个匹配到了原始，就算是原始正确；如果没有匹配到，说明原始标注错误；
对于小模型结果，标签错误、或者边界错误的，我们改了，需要跟原始Golden的做一下去重，去重后的，就是漏标注的, 看一下整体比例


先把gold相关的final结果放在一起，然后再和原始的gold来对比,需要获得最终人工标注正确的结果

'''
import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

def compare_lists_ignore_order(list1, list2):
    """比较两个列表是否包含相同的元素（不考虑顺序）"""
    if len(list1) != len(list2):
        return False
    
    # 对两个列表进行排序
    # 注意：字典需要有一个可比较的key，这里使用start, end和text的组合
    sorted_list1 = sorted(list1, key=lambda x: (x['start'], x['end'], x['text']))
    sorted_list2 = sorted(list2, key=lambda x: (x['start'], x['end'], x['text']))
    
    return sorted_list1 == sorted_list2
def merge_data(datalist, data_out):
    data_dic = {}
    for i in range(len(datalist)):
        temp = read_json(datalist[i])
        print(f"{datalist[i]}的论元个数为：{len(temp['annotations'])}")
        for ins in temp['annotations']:
            sen = ins['sentence']
            prd_word = ins['prd_word']
            prd_idx = ins['prd_idx']
            label = ins['label']
            key = "\t".join([sen, prd_word, str(prd_idx), label])
            if key not in data_dic.keys():
                data_dic[key] = ins
            else:
                if compare_lists_ignore_order(data_dic[key]['selected_spans'],ins['selected_spans']):
                    # print("Same!")
                    continue
                else:
                    print("error!")
                    import pdb;pdb.set_trace()
        
    print(len(data_dic))

    return data_dic


def get_all_orggold(org):
    org_dic = {}
    for i in range(len(org)):
        sen = org[i]['sen']
        prd_word = org[i]['prd_word']
        prd_idx = org[i]['prd_idx']
        label = org[i]['label']
        key = "\t".join([sen, prd_word, str(prd_idx), label])
        if key not in org_dic.keys():
            org_dic[key] = org[i]
        else:
            print("org error! have repeat")
            import pdb;pdb.set_trace()
    print(f"org data argument num: {len(org_dic)}")
    return org_dic


def compare_data_gold(org, fixed):
    same_key = 0
    right_key = 0
    wrong_key = 0
    extra = 0
    for ins in fixed.keys():
        flag = 0
        if ins in org.keys():
            same_key += 1
            span = org[ins]['span_idx'] # org span
            span_text = org[ins]['span']
            for exp in fixed[ins]['selected_spans']:
                if span == [exp["start"], exp['end']] and exp['text'] == span_text:
                    right_key+=1
                    flag = 1
                    break
            if flag == 0:
                wrong_key += 1
        else:
            extra += 1
            # print("多出来的！")
    
    right_org = len(org)-same_key+right_key
    print(f'right_key: {right_org}, all org: {len(org)}, {(len(org)-right_org)/len(org):.2%}')
    print(f"extra: {extra}")
    #178个是标注错误的

def compare_data_smallmodel(org, fixed):
    same_key = 0
    right_key = 0
    wrong_key = 0
    extra = 0
    for ins in fixed.keys():
        flag = 0
        if ins in org.keys():
            same_key += 1
            span = org[ins]['span_idx'] # org span
            span_text = org[ins]['span']
            for exp in fixed[ins]['selected_spans']:
                if span == [exp["start"], exp['end']] and exp['text'] == span_text:
                    right_key+=1
                    flag = 1
                    break
            if flag == 0:
                wrong_key += 1
        else:
            extra += 1
            # print("多出来的！")
    
    # right_org = same_key+right_key
    print(len(fixed))
    print(f'same_key: {same_key}, all org: {len(org)}, {same_key/len(org):.2%}')
    print(f"extra: {extra}")


if __name__=="__main__":
    domain = "bn"
    org_data = read_json(f"final_data/test_{domain}_4llm_core_gold.conll")
    gold_o1mini_right = "annotated_final/annotations_gold_o1right_random_wlj_final.json"
    gold_bothwrong = "annotated_final/annotations_single_gold_wlj_final.json"
    gold_o1wrong_dsright = "annotated_final/annotations_single_gold_random_wlj_final.json"
    data_list =[gold_o1mini_right, gold_bothwrong, gold_o1wrong_dsright]
    gold_repair = "annotated_final/allgold_final.json"
    
    
    fix_data_dic = merge_data(data_list, gold_repair)
    org_data_dic = get_all_orggold(org_data)
    compare_data_gold(org_data_dic, fix_data_dic)

    smallmodel_o1right_dswrong = "annotated_final/annotations_wlj_smallmodel_163_final.json"
    smallmodel_bothwrong = "annotated_final/annotations_smallmodel_botherror_random_wlj_final.json"
    data_list = [smallmodel_o1right_dswrong, smallmodel_bothwrong]
    smallmodel_repair = "annotated_final/allsmallmodel_final.json"
    fix_data_dic = merge_data(data_list, smallmodel_repair)
    compare_data_smallmodel(org_data_dic, fix_data_dic)
    

