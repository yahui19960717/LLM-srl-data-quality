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



def get_results():
    print(len(llmout), len(llmout2))
    for i in range(len(llmout2)):
        response, tag = parse_response(llmout2[i], llmout2[i]['response'])
        if tag:
            import pdb;pdb.set_trace()
        else:
            import pdb;pdb.set_trace()
   
# write_json(llmout, path_llmout)

if __name__ == "__main__":
    path_llmout = "llm/test_bn_4llm_all.conll"

    llmout = read_json(path_llmout)
    # llmout2 = read_json(path_llmout2)
    