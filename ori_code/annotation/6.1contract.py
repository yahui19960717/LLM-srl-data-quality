import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

def contract_twofile(file1, file2, wrong_file, right_file):

    # assert len(file1) == len(file2)
    wrong, right = 0, 0
    wrong_list, right_list = [], []
    print(len(file1), len(file2))
    for i in range(len(file1)):
        try:
            if file1[i]['span_mean']==None or file1[i]['label'] not in file1[i]['span_mean']:
                wrong+=1
                wrong_list.append(file2[i])
            else:
                if file1[i]['span_mean'][file1[i]['label']]!= file2[i]['span_mean']:
                    wrong+=1
                    wrong_list.append(file2[i])
                # import pdb;pdb.set_trace()
                else:
                    right+=1
                    right_list.append(file2[i])
        except:
            import pdb;pdb.set_trace()
    
    print(wrong, right)
    write_json(right_list, right_file)
    write_json(wrong_list, wrong_file)
    assert len(right_list)+len(wrong_list) == len(file1)


if __name__=="__main__":
    domain="bn"

    file1 = read_json(f"final_data/errorversion_test_bn_4llm_core_gold.conll")
    file2 = read_json(f"final_data/test_bn_4llm_core_gold.conll")
    file1sense = f"llm-temp/sensewrong_test_bn_4llm_core_gold.conll"
    file2sense = f"llm-temp/senseright_test_bn_4llm_core_gold.conll"
     # sense
    contract_twofile(file1, file2, file1sense, file2sense)
    # file1 = read_json(f"final_data/errorversion_test_bn_4llm_core_smallmodel.conll")
    # file2 = read_json(f"final_data/test_bn_4llm_core_smallmodel.conll")
    # file1sense = f"llm-temp/sensewrong_test_bn_4llm_core_smallmodel.conll"
    # file2sense = f"llm-temp/senseright_test_bn_4llm_core_smallmodel.conll"
    # contract_twofile(file1, file2, file1sense, file2sense)

