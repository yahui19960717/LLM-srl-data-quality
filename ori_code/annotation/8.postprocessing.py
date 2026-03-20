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

def parse_response(instance, response):
    try:
        parse_tag = False
        res_str = response.strip()
        if not res_str:
            print("Error:empty response!")
            return response, parse_tag
        if res_str.startswith("```json"):
            res_str = res_str[7:]
        if res_str.endswith("```"):
            res_str = res_str[:-3]
        res_str = res_str.strip()

        res_data = json.loads(res_str)
        parse_tag = True
        res_data['idx'] = instance['idx']
        res_data['sen'] = instance['sen']
        res_data['prd_word'] = instance['prd_word']
        res_data['span'] = instance['span']
        res_data['prd_lemma'] = instance['prd_lemma']
        res_data['prd_sense'] = instance['prd_sense']
        res_data['prd_idx'] = instance['prd_idx']
        res_data['label'] = instance['label']
        res_data['span_idx'] = instance['span_idx']
        return res_data, parse_tag
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag
    except Exception as e:
        print(f"其他错误: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag



def get_results(llmout, incorrect_file, correct_file):
    
    count_incorrect, count_correct = 0, 0 
    incorrect_data, correct_data = [], []
    for i in range(len(llmout)):
        response, tag = parse_response(llmout[i], llmout[i]['response'])
        if tag:
            idx = llmout[i]['idx']
            sen = llmout[i]['sen']
            prd_word = llmout[i]['prd_word']
            span = llmout[i]['span']
            prd_lemma = llmout[i]['prd_lemma']
            prd_sense = llmout[i]['prd_sense']
            prd_idx = llmout[i]['prd_idx']
            label = llmout[i]['label']
            span_idx = llmout[i]['span_idx']
            span_mean = llmout[i]['span_mean']
            try:    
                final_judgement = response['final_judgement']
            except:
                incorrect_data.append(llmout[i])
                print(0)
            if final_judgement == "incorrect":
                count_incorrect += 1
                incorrect_data.append(llmout[i])
            else:
                count_correct += 1
                correct_data.append(llmout[i])
        else:
            print(1)
            # import pdb;pdb.set_trace()
        

    write_json(correct_data, correct_file)
    write_json(incorrect_data, incorrect_file)
    print(f'correct: {len(correct_data)}, incorrect: {len(incorrect_data)}, all: {len(correct_data)+ len(incorrect_data)}')

if __name__ == "__main__":
    # gold results
    domain = "bn"




    
    # 可标可不标注补充
    path_llmout = f"llm/test_{domain}_4llm_all_smnotrecall.conll"
    path_llmout2 = f"llm/test_{domain}_4llm_parseright_smnotrecall.conll"
    llmout = read_json(path_llmout)
    llmout2 = read_json(path_llmout2)
    print(len(llmout), len(llmout2))
    incorrect_data, correct_data = f"llm/incorrect_data_{domain}_smnotrecall.json", f"llm/correct_data_{domain}_smnotrecall.json"
    get_results(llmout, incorrect_data, correct_data) # 正确247 ,不正确的34

    # path_llmout = "llm/test_bn_4llm_all_gold.conll"
    # path_llmout2 = "llm/test_bn_4llm_parseright_gold.conll"
    # llmout = read_json(path_llmout)
    # llmout2 = read_json(path_llmout2)
    # print(len(llmout), len(llmout2))
    # incorrect_data, correct_data = f"llm/incorrect_data_{domain}_gold.json", f"llm/correct_data_{domain}_gold.json"
    # get_results(llmout, incorrect_data, correct_data)



    # small model predictions
    # domain = "bn"
    # path_llmout = "llm/test_bn_4llm_all_smallmodel.conll"
    # path_llmout2 = "llm/test_bn_4llm_parseright_smallmodel.conll"
    # llmout = read_json(path_llmout)
    # llmout2 = read_json(path_llmout2)
    # print(len(llmout), len(llmout2))
    # incorrect_data, correct_data = f"llm/incorrect_data_{domain}_smallmodel.json", f"llm/correct_data_{domain}_smallmodel.json"
    # get_results(llmout, incorrect_data, correct_data)