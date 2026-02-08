from config import Defination, read, write_json, read_json, labels_conll
import json

data = read_json("/data/ljwang/span-SRL-LLM/llmout_lyh/nw/temp.json")
all_data = []
for instance in data:

    response = instance['response']
    # import pdb;pdb.set_trace()
    print(instance['index_sen'])
    res_str = response.strip()
    if not res_str:
        print("Error:empty response!")
    if res_str.startswith("```json"):
        res_str = res_str[7:]
    if res_str.endswith("```"):
        res_str = res_str[:-3]
    res_str = res_str.strip()

    res_data = json.loads(res_str)
    res_data['index_sen'] = instance['index_sen']
    res_data['sentences'] = instance['sentences']
    res_data['predicate'] = instance['predicate']
    res_data['selected_span'] = instance['selected_span']
    res_data['error_type'] = instance['error_type']
    res_data['gold_label'] = instance['gold_label']
    res_data['org_span'] = instance['org_span']
    res_data['conflict_span'] = instance['conflict_span']
    all_data.append(res_data)
write_json(all_data, "/data/ljwang/span-SRL-LLM/llmout_lyh/nw/temp-v1.json")
    