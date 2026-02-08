# 给定句子和自然语言问题，构建LLM prompt来回答自然语言问题，

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
import time
from collections import defaultdict, Counter
from config import Defination, read, write_json, read_json, labels_conll

random.seed(42)
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",#tcy
    #api_key="sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz" # gpt-4o-mini
    #api_key="sk-wUaE2guPPNKLKYTWAhcq7pFNXQMQa6JadrASpFdxsPBbKgkm" # o1-mini
    #api_key="sk-uwG3FOUM8NQJswVXoWOLuVYa7OrDoNxd7t7O1it5RLirJiMP" # deepseek-v3.2
    #api_key="sk-wUaE2guPPNKLKYTWAhcq7pFNXQMQa6JadrASpFdxsPBbKgkm" # DeepSeek-V3.2另一个key
    api_key="sk-7k63FEwvaVFJnUD91JNozH9LUDHoeiH5MCb8Hjt9a5I9UY04"
)


# 根据给定的句子和自然语言问题，判断自然语言问题是否可回答并从句子中抽取其答案
def get_prompt_for_question_answer(sentence, natural_question):
    prompt = f""" You are a professional linguistic analysis expert. Your core task is to extract the accurate answer to a given natural language question exclusively from the corresponding provided sentence. \
    \n ### Task Execution Steps \
    \n 1. First, judge whether the answer to the natural language question can be found in the given sentence based on the sentence and the question itself. If the answer is unavailable, output no answer. \
    \n 2. If the answer to the natural language question can be found in the sentence, extract the answer verbatim from the sentence and output it. \
    \n ### Output Requirements: \
    \n You must strictly output the final result in standard JSON format, which contains the following two mandatory fields with fixed rules: \
    \n - has_answer: The value is limited to yes or no, indicating whether the natural language question has a retrievable answer in the sentence. \
    \n - answer: The specific answer extracted from the sentence for the natural language question; if there is no available answer, this field must be left empty. \
    \n ### Example: \
    \n **Input**: \
    \n - Sentence: UCD finished the 2006 championship as Dublin champions , by beating St Vincents in the final . \
    \n - Natural language question: Who finished the 2006 championship? \
    \n **Output**: {{"has_answer": "yes", "answer": "UCD"}} \
    \n \
    \n **Input**: \
    \n - Sentence: UCD finished the 2006 championship as Dublin champions , by beating St Vincents in the final . \
    \n - Natural language question: What did UCD finish? \
    \n **Output**: {{"has_answer": "yes", "answer": "the 2006 championship"}} \
    \n \
    \n **Input**: \
    \n - Sentence: UCD finished the 2006 championship as Dublin champions , by beating St Vincents in the final .\
    \n - Natural language question: How did UCD finish the 2006 championship? \
    \n **Output**: {{"has_answer": "yes", "answer": "by beating St Vincents in the final"}} \
    \n \
    \n **Input**: \
    \n - Sentence: %hm . we 'll see what happens . \
    \n - Natural language question: When we see? \
    \n **Output**: {{"has_answer": "no", "answer": ""}} \
    \n \
    \n **Input**: \
    \n - Sentence: Yes she did . \
    \n - Natural language question: What did she do? \
    \n **Output**: {{"has_answer": "no", "answer": ""}} \
    \n ### Input \
    \n - Sentence: {sentence} \
    \n - Natural language question: {natural_question} \
    \n \
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
    llm_response = instance['response'] if 'response' in instance else "" #llm预测结果
    #print(f'{llm_response}')
    if isinstance(llm_response, dict) == False:
        #print("llm_response不是字典类型！")
        llm_response = json.loads(llm_response)
    generate_question = llm_response['question'] if 'question' in llm_response else "" #llm生成的问题
    has_question = llm_response['exist_relation'] if 'exist_relation' in llm_response else "" #llm是否生成问题
    #print(f'question: {generate_question}, has_question:{has_question}')

    flag = 1 # 生成结果并回答
    if has_question == "yes" and generate_question.strip() != "":
        prompt = get_prompt_for_question_answer(sentence, generate_question)
        return prompt, flag
    else:
        flag = 0 #没有生成问题，直接返回没有答案
        prompt = {"has_answer": "no", "answer": ""}
        return prompt, flag

def get_completion(prompt):
    response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3.2",
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
                extra_body={"thinking": {"type": "enabled"}}
            )
    
    res = response.choices[0].message.content
    return res
    
def parse_response(instance, response):
    try:
        parse_tag = False
        
        if isinstance(response, dict):
            res_data = response
        else:
            res_str = response.strip()
            if not res_str:
                print("Error:empty response!")
                return response, parse_tag
            
            # 尝试提取JSON部分
            start_idx = res_str.find('{')
            end_idx = res_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                res_str = res_str[start_idx:end_idx+1]
            else:
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
        res_data['exist_relation'] = instance.get('exist_relation', "")
        res_data['question'] = instance.get('question', "")
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
    total_num = 0
    correct_num = 0
    ori_correct_num = 0
    core_total_num = 0
    core_correct_num = 0
    ori_core_correct_num = 0
    nocore_total_num = 0
    nocore_correct_num = 0
    ori_nocore_correct_num = 0
    for instance in data:
        has_answer = instance.get('has_answer', None)
        answer = instance.get('answer', '').strip()
        select_span = instance.get('selected_span', None)
        error_type = instance.get('error_type', None)
        temp_span = select_span[1].split("-")[-1]

        total_num += 1
        if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
            core_total_num += 1
        else:
            nocore_total_num += 1

        if error_type == 'right':
            ori_correct_num += 1
            if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
                ori_core_correct_num += 1
            else:
                ori_nocore_correct_num += 1
            if select_span[0] == answer:
                correct_num += 1
                if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
                    core_correct_num += 1
                else:
                    nocore_correct_num += 1
        else:
            if has_answer == "no" or answer == "" or answer != select_span[0]:
                correct_num += 1
                if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
                    core_correct_num += 1
                else:
                    nocore_correct_num += 1

    total_acc = correct_num / total_num if total_num > 0 else 0
    ori_acc = ori_correct_num / total_num if total_num > 0 else 0
    ori_core_acc = ori_core_correct_num / core_total_num if core_total_num > 0 else 0
    ori_nocore_acc = ori_nocore_correct_num / nocore_total_num if nocore_total_num > 0 else 0
    core_acc = core_correct_num / core_total_num if core_total_num > 0 else 0
    nocore_acc = nocore_correct_num / nocore_total_num if nocore_total_num > 0 else 0
    print(f"原始准确率：{ori_acc:.4f}, 原始核心论元准确率：{ori_core_acc:.4f}, 原始非核心论元准确率: {ori_nocore_acc:.4f}")
    print(f"LLM准确率: {total_acc:.4f}，LLM核心论元准确率: {core_acc:.4f}，LLM非核心论元准确率: {nocore_acc:.4f}")
    
    return total_acc

def LLM_prompt(data, path_save, path_save2):
    # openai.api_timeout = 60

    results = []
    results_simple = []

    if os.path.exists(path_save):
        results = read_json(path_save)
        final_ins = results[-1]
        for i in range(len(data)):
            if data[i]["index_sen"] == final_ins['index_sen']:
                data = data[i+1:]       
                break
    if os.path.exists(path_save2):
        results_simple = read_json(path_save2)
    print(f'已存在的数据有:{len(results)}')  
    print("-" * 60)

    test_num = 0
    for instance in tqdm(data): # 一个句子
        #if test_num > 10:
        #    break
        #test_num += 1
        prompt, res_flag = build_prompt(instance)
        if res_flag == 1:
            res =  get_completion(prompt)
        else:
            res = prompt
        instance['ans_response'] = res
        results.append(instance)

        res_data, res_tag = parse_response(instance, res)
        if res_tag == True:
            results_simple.append(res_data)
        time.sleep(10)  # 添加延时，避免请求过于频繁

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
                data = read_json(f'../llmout_question_generation/{i}/{i}-{j}-{k}-ds.json')
                path_llmout = f'../llmout_question_answer/{i}/{i}-{j}-{k}-ds-o1.json'
                path_llmout2 = f'../llmout_question_answer/{i}/{i}-{j}-{k}-ds-o1-simp.json'
                LLM_prompt(data, path_llmout, path_llmout2)
                print("工作保存完成！")
