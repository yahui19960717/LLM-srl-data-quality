# 针对句子、谓词和角色关系，生成以角色关系对应论元为答案的问题，LLM生成3-5个问题

import os
import sys
import time
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
    #api_key="sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz" # o1-mini
    #api_key="sk-uwG3FOUM8NQJswVXoWOLuVYa7OrDoNxd7t7O1it5RLirJiMP" # deepseek-v3.2
    api_key="sk-7k63FEwvaVFJnUD91JNozH9LUDHoeiH5MCb8Hjt9a5I9UY04" # lijie
)

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

def get_question_from_general(new_sen, predicate, role):
    question = ""
    if role == 'ARG0':
        question = "what is the subject or the initiator of " + predicate + " ?"
    elif role == 'ARG1':
        question = "what is patient or theme or content of " + predicate + " ?"
    elif role == 'ARG2':
        question = "what is instrument or benefactive or attribute of " + predicate + " ?"
    elif role == 'ARG3':
        question = "what is starting point of " + predicate + " ?"
    elif role == 'ARG4':
        question = "what is ending point of " + predicate + " ?"
    return question


def get_question_from_framefile(new_sen, predicate, role, role_mean):
    question = "what is " + role_mean + " of " + predicate + " ?"
    return question


#如果有framefile就使用framefile中定义的角色+模版，否则就使用通用的问题模版
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

    flag = 0
    temp_span = sen_span[1].split("-")[-1] # 取最后，不考虑R-，C-
    if error_type == 'label_error' and gold_label != "":
        temp_span = gold_label.split("-")[-1]
    if temp_span not in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}:
        return "", "", flag
    
    
    #获取candidate中的角色定义，基于模版进行提问；如果没有candidate，则使用通用的问题模版；
    generate_question = ""
    try:
        candidate_labels = instance['candidate_roles']
        if candidate_labels == {}:
            generate_question = get_question_from_general(new_sen_, predicate, temp_span)
        elif temp_span not in candidate_labels or candidate_labels[temp_span] == "":
            generate_question = get_question_from_general(new_sen_, predicate, temp_span)
        else:
            generate_question = get_question_from_framefile(new_sen_, predicate, temp_span, candidate_labels[temp_span])
    except:
        generate_question = get_question_from_general(new_sen_, predicate, temp_span)
    if generate_question == "":
        return "", "", flag
    flag = 1
    answer_prompt = get_prompt_for_question_answer(new_sen_, generate_question) 
    return answer_prompt, generate_question, flag

def get_completion(prompt):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="o1-mini", #"deepseek-ai/DeepSeek-V3.2",
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
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"All {max_retries} attempts failed.")
                return ""

    
def parse_response(instance, response, question):
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
        res_data['generate_question'] = question
        return res_data, parse_tag
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag
    except Exception as e:
        print(f"其他错误: {str(e)}，response原始内容: {response}")
        parse_tag = False
        return response, parse_tag

def LLM_prompt(data, path_save, path_save2):
    # openai.api_timeout = 60

    results = []
    results_simple = []

    test_num = 0
    for instance in tqdm(data): # 一个句子
        #if test_num > 10:
        #    break
        #test_num += 1
        prompt, question, flag = build_prompt(instance)
        if flag == 0:
            continue #不处理非核心角色
        res =  get_completion(prompt)
        res_data, res_tag = parse_response(instance, res, question)
        if res_tag == True:
            results_simple.append(res_data)
        instance['response'] = res
        instance['generate_question'] = question
        results.append(instance)

    json.dump(results, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    json.dump(results_simple, open(path_save2, 'w', encoding="utf-8"), indent=0, ensure_ascii=False)


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU

    data = read_json(f'../forllm_frames_newest/nw/nw-bn-test.json')
    path_llmout = f'question_generation_result/nw-bn-test-pattern-o1.json'
    path_llmout2 = f'question_generation_result/nw-bn-test-pattern-o1-simp.json'
    LLM_prompt(data, path_llmout, path_llmout2)
    
