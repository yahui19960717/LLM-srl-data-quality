# 给定句子、谓词和论元，构建LLM prompt来生成自然语言问题，

import os
import sys
from sys import excepthook
from sympy import O
import torch
from tqdm import tqdm
import json
from openai import OpenAI
import numpy as np
import random
from collections import defaultdict, Counter
from config import Defination, read, write_json, read_json, labels_conll

random.seed(42)
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",#tcy
    #api_key="sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz" # gpt-4o-mini
    api_key="sk-wUaE2guPPNKLKYTWAhcq7pFNXQMQa6JadrASpFdxsPBbKgkm" # o1-mini
    #api_key="sk-uwG3FOUM8NQJswVXoWOLuVYa7OrDoNxd7t7O1it5RLirJiMP" # deepseek-v3.2
)


# 根据给定的句子、谓词和论元，判断谓词和论元是否有关系，并生成以论元为答案的自然语言问题
def get_prompt_for_question_generation(sentence, predicate, argument):
    prompt = f""" You are a linguistic analysis expert and are now required to generate a natural language question with the argument as the answer based on the given sentence, predicate and argument. \
    \n ### Task Description \
    \n Natural language question generation refers to generating a natural language question with the argument as the answer according to the given sentence, predicate and argument. The generated question must accurately reflect the semantic role of the argument in the sentence, and should feature a clear, concise description, strict grammatical compliance and definite answerability. \
    \n If the given argument and predicate have no syntactic or semantic relationship, making it impossible to generate a valid and reasonable question, no corresponding question needs to be generated for such cases. \
    \n ### Task Execution Steps \
    \n 1. Read and comprehend the sentence carefully, then judge whether there is a syntactic and semantic relationship between the given predicate and argument. Skip the question generation step if no such relationship exists. \
    \n 2. Select an appropriate interrogative word (e.g., who, what, where, when, how) based on the semantic role of the argument, and construct a grammatically compliant natural language question to ensure that the question can be answered with the given argument. \
    \n ### Output Requirements: \
    \n Output the result in strict JSON format with the following two fields: \
    \n - exist_relation: the value is either "yes" or "no", indicating whether a syntactic and semantic relationship exists between the predicate and the argument; \
    \n - question: the string of the generated natural language question; leave this field empty if there is no relationship between the predicate and the argument. \
    \n ### Input Description: \
    \n The input consists of a sentence, a predicate and an argument. The predicate is enclosed in underscores (i.e., "_"), and the argument is marked in bold (i.e., "**"). Note that the sentence may contain the same words as the predicate or argument in different positions; you must accurately identify the target predicate and argument strictly according to the underscore and bold marks. \
    \n ### Example: \
    \n **Input**: \
    \n - Sentence: **UCD** _finished_ the 2006 championship as Dublin champions, by beating St Vincents in the final. \
    \n - Predicate: finished \
    \n - Argument: UCD \
    \n **Output**: {{"exist_relation": "yes", "question": "Who finished the 2006 championship?"}} \
    \n \
    \n **Input**: \
    \n - Sentence: UCD _finished_ **the 2006 championship** as Dublin champions, by beating St Vincents in the final. \
    \n - Predicate: finished \
    \n - Argument: the 2006 championship \
    \n **Output**: {{"exist_relation": "yes", "question": "What did UCD finish?"}} \
    \n \
    \n **Input**: \
    \n - Sentence: UCD _finished_ the 2006 championship as Dublin champions, **by beating St Vincents in the final**. \
    \n - Predicate: finished \
    \n - Argument: by beating St Vincents in the final \
    \n **Output**: {{"exist_relation": "yes", "question": "How did UCD finish the 2006 championship?"}} \
    \n \
    \n **Input**: \
    \n - Sentence: UCD finished the 2006 championship as Dublin champions, by _beating_ St Vincents **in the final**. \
    \n - Predicate: beating \
    \n - Argument: in the final \
    \n **Output**: {{"exist_relation": "yes", "question": "When did someone beat St Vincents?"}} \
    \n \
    \n **Input**: \
    \n - Sentence: like mom said she said even if it was going to be in September , she said the little things I can tolerate you know if **that** happens you _know_ . \
    \n - Predicate: know \
    \n - Argument: that \
    \n **Output**: {{"exist_relation": "no", "question": ""}} \
    \n ### Input \
    \n\t Sentence: {sentence} \
    \n\t Predicate: {predicate} \
    \n\t Argument: {argument} \
    \n Please Answer only with a JSON object, without any additional text, explanations, or Markdown formatting. \
    """
    return prompt   


# flag=0说明谓词没有论元；flag=1说明没有这个标签；flag=2说明没有framefile
def build_prompt(instance):
    sen_id = instance['index_sen'] #句子id
    sentence = instance['sentences'] #句子字符串
    predicate = instance['predicate'] #给定的谓词字符串
    sen_span = instance['selected_span'] #候选论元信息 (含论元字符串, 角色标签, 论元在句子中的位置，起始位置、终止位置+1)
    error_type = instance['error_type'] #错误类型，根据golden结果和预测结果生成的，含关系错误、边界错误、正确、多余
    gold_label = instance['gold_label'] #正确角色关系标签
    org_span = instance["org_span"] # golden论元在句子xx，谓词id+1，起始位置、终止位置+1，xx
    prd_id = org_span[1]
    prd_with_ = "_"+predicate+"_" # 谓词加下划线标记，标记谓词，防止多个相同词有歧义
    arg_with_star = "**"+sen_span[0]+"**" # 论元加粗标记，防止多个相同词有歧义

    #构建新的句子，把谓词用下划线标记，论元用加粗标记
    new_sen = sentence.strip().split().copy()
    new_sen[prd_id-1]=prd_with_
    new_sen = new_sen[0:sen_span[2][0]]+ [arg_with_star] + new_sen[sen_span[2][1]:]
    new_sen_ = " ".join(new_sen)
    #print(new_sen)

    prompt = get_prompt_for_question_generation(new_sen_, predicate, sen_span[0])
    return prompt

def get_completion(prompt):
    response = client.chat.completions.create(
                model= "o1-mini", #"deepseek-ai/DeepSeek-V3.2",
                messages=[
                    {
                        "role": "system",
                        "content":"You are an expert Semantic Role Labeling Quality Assurance Inspector. "
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                #extra_body={"thinking": {"type": "enabled"}}
            )
    
    res = response.choices[0].message.content
    return res
    
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
        res_data['index_sen'] = instance['index_sen']
        res_data['sentences'] = instance['sentences']
        res_data['predicate'] = instance['predicate']
        res_data['selected_span'] = instance['selected_span']
        res_data['predict_prob'] = instance['predict_prob']
        res_data['error_type'] = instance['error_type']
        res_data['gold_label'] = instance['gold_label']
        res_data['org_span'] = instance['org_span']
        return res_data, parse_tag
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag
    except Exception as e:
        print(f"其他错误: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag

def stas_accuracy(data):
    redundant_num = 0
    redundant_gen_num = 0
    total_num = 0
    total_gen_num = 0
    for instance in data:
        gen_flag = instance.get('exist_relation', None)
        question = instance.get('question', '').strip()
        error_type = instance.get('error_type', None)
        if error_type == 'redundant':
            redundant_num += 1
            if gen_flag == 'no' or question == '':
                redundant_gen_num += 1
        else:
            total_num += 1
            if gen_flag == 'yes' and question != '':
                total_gen_num += 1

    redundant_accuracy = redundant_gen_num / redundant_num if redundant_num > 0 else 0
    total_gen_ratio = total_gen_num / total_num if total_num > 0 else 0
    print(f"Total redundant instances: {redundant_num}, Correctly not generated: {redundant_gen_num}, Redundant Generation Accuracy: {redundant_accuracy:.4f}")
    print(f"Total non-redundant instances: {total_num}, Correctly generated: {total_gen_num}, Generation Ratio: {total_gen_ratio:.4f}")
    
    return redundant_accuracy

def LLM_prompt(data, path_save, path_save2):
    # openai.api_timeout = 60

    results = []
    results_simple = []

    test_num = 0
    for instance in tqdm(data): # 一个句子
        #if test_num > 10:
        #    break
        #test_num += 1
        prompt = build_prompt(instance)
        res =  get_completion(prompt)
        instance['response'] = res
        results.append(instance)

        res_data, res_tag = parse_response(instance, res)
        if res_tag == True:
            results_simple.append(res_data)

    json.dump(results, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    json.dump(results_simple, open(path_save2, 'w', encoding="utf-8"), indent=0, ensure_ascii=False)
    
    redundanct_acc = stas_accuracy(results_simple)

# 训练遍历TEST下每一个文件，针对每一个边缘概率对应角色进行问题生成
if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU

    dataset = ['test'] #dev, 
    source = ['nw'] #["nw",  "bn", "bc" ]
    target = ['tc'] #['tc', 'bn', 'nw', 'bc']
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                data = read_json(f'../forllm_frames_newest/{i}/{i}-{j}-{k}.json')
                path_llmout = f'../llmout_question_generation/{i}/{i}-{j}-{k}-o1.json'
                path_llmout2 = f'../llmout_question_generation/{i}/{i}-{j}-{k}-o1-simp.json'
                LLM_prompt(data, path_llmout, path_llmout2)
                print("工作保存完成！")
