# 构建prompt来应用LLM

import os
import sys
#print(sys.executable)
from sys import excepthook
from sympy import O
import torch
#print(torch.__file__)
from tqdm import tqdm
import json
from openai import OpenAI
import numpy as np
import random
from collections import defaultdict, Counter
from config import Defination, read, write_json, read_json, labels_conll
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']

random.seed(42)
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",#tcy
    api_key="sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz" # tcy
)


def check_error(instance):
    sen_id = instance['index_sen'] #句子id
    sentence = instance['sentences'] #句子字符串
    sen_span = instance['selected_span'] #候选论元信息 (含论元字符串, 角色标签, 论元在句子中的位置，起始位置、终止位置+1)
    error_type = instance['error_type'] #错误类型，根据golden结果和预测结果生成的，含关系错误、边界错误、正确、多余
    org_span = instance["org_span"]  # predict论元在句子xx，谓词id+1，起始位置、终止位置+1，xx


    if error_type == "boundary_error":
        if sen_span[2][0] == org_span[2] and sen_span[2][1] == org_span[3]:  #预测边界跟golden边界一致，应该是数据处理错误
            return True
    return False

def check_data(data, path_save):
    
    error_instance_list = []
    error_num = 0
    for instance in tqdm(data): 
        flag = check_error(instance)
        if flag == True:
            error_num += 1
            error_instance_list.append(instance)
    print(error_num)
    json.dump(error_instance_list, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 



#核心角色，且知道该谓词frame中有这个角色，重点判断角色是否提取正确
def get_prompt_for_corerole(sentence, predicate, argument, role, role_mean):
    prompt = f""" You are a professional linguistic analysis expert and are now required to perform the semantic role labeling verification task. \
    \n Your task is to evaluate the correctness of the argument and that of the semantic role label for the given sentence, predict, argument and semantic role label. If any errors are found, you must provide a detailed explanation. and provide a revised solution simultaneously \
    \n #### Fundamental Definition of SRL:  \
    \n Semantic Role Labeling is a shallow semantic parsing task. Its core objective is to identify the arguments (i.e., participants or states) related to a given predicate and assign them specific semantic roles (e.g., Agent, Patient, Time, Location, etc.). Both the predicate and its arguments are specific words or phrases extracted directly from the sentence. \
    \n **Predicate**: refering to the word or phrase in a sentence that expresses the core action, behavior, state, relationship, or attribute. There are four categories of predicates: \
    \n\t 1. verbal predicates: they express concrete actions or behaviors in a setence, clause or phrase; \
    \n\t 2. nominal predicates: nouns that express core events or relationships can function as predicates, for example: “The committee announced the appointment of a new director.” -> “appointment” is a noun predicate; \
    \n\t 3. adjectival predicates: they express the attributes, characteristics, or states of entities and serve as the core meaning in a sentence or clause; \
    \n\t 4. stative verbs/copular verbs predicates: they express the subject's state, attributes, existence, or possession, without involving a concrete action, but serve as the core meaning of the sentence or clause.\
    \n **Argument**: refering to the word or phrase in a sentence that plays a specific role in relation to the predicate, such as agent, patient, instrument, location, time, etc. Arguments are essential components that help complete the meaning of the predicate and provide additional information about the event or state described by the predicate. \
    \n #### Execution Steps and Output Requirements: \
    \n 1. Argument Extraction Verification: 
    \n\t - Determine whether the given argument is correctly extracted based on syntactic and semantic information. The possible results are: boundary correct, insufficient extraction, over-extraction, boundary incorrect. In this task, according to the "{sentence}" and the "{predicate}", based on syntactic and semantic knowledge, you need to determine whether the provided argument span is correct. \
    \n\t - If the provided argument span is not extracted correctly, please provide the correct argument span. \
    \n\t - Output in JSON format with the key fields being argument_extraction_evaluation, reason_for_judgment, and correct_argument. For the filed of argument_extraction_evaluation, the candidate value is "yes" or "no". If the provided argument span is correct, the correct_argument filed if left empty. \
    \n 2. Semantic Role Correctness Validation: \
    \n\t - Determine whether the argument and the predicate satisfy the specified role relationship. The primary results include: correct relation and incorrect relation. In this task, if the "{argument}" is the "{role_mean}" of the "{predicate}", then they satisfy the relation of "{role}", and the reverse is not true.\
    \n\t - If the argument and predicate do not satisfy the specified role relationship, please provide the correct semantic role label. \
    \n\t - Output in JSON format with the key fields being role_evaluation, reason_for_judgment, and correct_role. For the filed of role_evaluation, the candidate value is "yes" or "no". If the role is correct, the correct_role field is left empty. \
    \n 3.  Summary of Results: \
    \n\t - According to the results of the argument extraction evaluation and the semantic role correctness validation, provide a summary judgment. If the fields of argument_extraction_evaluation and role_evaluation are both "yes", the final result is correct; if even one value is not "yes", the final result will be incorrect. \
    \n\t - Output in JSON format with the key fields being final_judgment and reason_for_judgment. The candidate values for final_judgment are "correct" and "incorrect".  \
    \n #### Input Specification \
    \n We will input a sentence, a predicate, an argument, a role label and the description of role label. Both the predicate and the argument are words or word fragments from the input sentence. Since there may be multiple identical words in the sentence but at different positions, to better distinguish which word is being referred to, the predicate in the input is marked with underscores (i.e., wrapped with _), and the semantic role field is highlighted in bold (i.e., wrapped with **). \
    \n #### Input \
    \n\t Sentence: {sentence} \
    \n\t Predicate: {predicate} \
    \n\t Argument: {argument} \
    \n\t Role label: {role} \
    \n\t Role description: {role_mean} \
    \n Please Answer only with a JSON object. \
    """
    return prompt

#非核心角色，重点判断角色是否提取正确、以及该谓词是否有这个角色
def get_prompt_for_otherrole(sentence, predicate, argument, role, role_mean):
    prompt = f""" You are a professional linguistic analysis expert and are now required to perform the semantic role labeling verification task. \
    \n Your task is to evaluate the correctness of the argument and that of the semantic role label for the given sentence, predict, argument and semantic role label. If any errors are found, you must provide a detailed explanation. and provide a revised solution simultaneously \
    \n #### Fundamental Definition of SRL:  \
    \n Semantic Role Labeling is a shallow semantic parsing task. Its core objective is to identify the arguments (i.e., participants or states) related to a given predicate and assign them specific semantic roles (e.g., Agent, Patient, Time, Location, etc.). Both the predicate and its arguments are specific words or phrases extracted directly from the sentence. \
    \n **Predicate**: refering to the word or phrase in a sentence that expresses the core action, behavior, state, relationship, or attribute. There are four categories of predicates: \
    \n\t 1. verbal predicates: they express concrete actions or behaviors in a setence, clause or phrase; \
    \n\t 2. nominal predicates: nouns that express core events or relationships can function as predicates, for example: “The committee announced the appointment of a new director.” -> “appointment” is a noun predicate; \
    \n\t 3. adjectival predicates: they express the attributes, characteristics, or states of entities and serve as the core meaning in a sentence or clause; \
    \n\t 4. stative verbs/copular verbs predicates: they express the subject's state, attributes, existence, or possession, without involving a concrete action, but serve as the core meaning of the sentence or clause.\
    \n **Argument**: refering to the word or phrase in a sentence that plays a specific role in relation to the predicate, such as agent, patient, instrument, location, time, etc. Arguments are essential components that help complete the meaning of the predicate and provide additional information about the event or state described by the predicate. \
    \n #### Execution Steps and Output Requirements: \
    \n 1. Argument Extraction Verification: 
    \n\t - Determine whether the given argument is correctly extracted based on syntactic and semantic information. The possible results are: boundary correct, insufficient extraction, over-extraction, boundary incorrect. In this task, according to the "{sentence}" and the "{predicate}", based on syntactic and semantic knowledge, you need to determine whether the provided argument span is correct. \
    \n\t - If the provided argument span is not extracted correctly, please provide the correct argument span. \
    \n\t - Output in JSON format with the key fields being argument_extraction_evaluation, reason_for_judgment, and correct_argument. For the filed of argument_extraction_evaluation, the candidate value is "yes" or "no". If the provided argument span is correct, the correct_argument filed if left empty. \
    \n 2. Semantic Role Correctness Validation: \
    \n\t - Determine whether the argument and the predicate satisfy the specified role relationship. The primary results include: correct relation and incorrect relation. In this task, if the "{argument}" indicates the "{role_mean}" in which the "{predicate}" takes place, then they satisfy the relation of "{role}", and the reverse is not true. Please note that the given argument and predicate may be completely unrelated, or the predicate may not have an argument corresponding to this role relation.\
    \n\t - If the argument and predicate do not satisfy the specified role relationship, but they have other relationship, please provide the correct semantic role label. \
    \n\t - Output in JSON format with the key fields being role_evaluation, reason_for_judgment, and correct_role. For the filed of role_evaluation, the candidate value is "yes" or "no". If the role is correct, the correct_role field is left empty. If the argument and the predicate are completely unrelated, the correct_role filed is filled with Null. \
    \n 3.  Summary of Results: \
    \n\t - According to the results of the argument extraction evaluation and the semantic role correctness validation, provide a summary judgment. If the fields of argument_extraction_evaluation and role_evaluation are both "yes", the final result is correct; if even one value is not "yes", the final result will be incorrect. \
    \n\t - Output in JSON format with the key fields being final_judgment and reason_for_judgment. The candidate values for final_judgment are "correct" and "incorrect".  \
    \n #### Input Specification \
    \n We will input a sentence, a predicate, an argument, a role label and the description of role label. Both the predicate and the argument are words or word fragments from the input sentence. Since there may be multiple identical words in the sentence but at different positions, to better distinguish which word is being referred to, the predicate in the input is marked with underscores (i.e., wrapped with _), and the semantic role field is highlighted in bold (i.e., wrapped with **). \
    \n #### Input \
    \n\t Sentence: {sentence} \
    \n\t Predicate: {predicate} \
    \n\t Argument: {argument} \
    \n\t Role label: {role} \
    \n\t Role description: {role_mean} \
    \n Please Answer only with a JSON object. \
    """
    return prompt



# flag=0说明谓词没有论元；flag=1说明没有这个标签；flag=2说明没有framefile
def build_prompt(instance):
    sen_id = instance['index_sen'] #句子id
    sentence = instance['sentences'] #句子字符串
    predicate = instance['predicate'] #给定的谓词字符串
    sen_span = instance['selected_span'] #候选论元信息 (含论元字符串, 角色标签, 论元在句子中的位置，起始位置、终止位置+1)
    predict_prob = instance['predict_prob'] #模型预测的概率（用于选择边缘概率）
    error_type = instance['error_type'] #错误类型，根据golden结果和预测结果生成的，含关系错误、边界错误、正确、多余
    gold_label = instance['gold_label'] #正确角色关系标签
    lemma = instance['lemma'] #谓词lemma，原型
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

    try:
        candidate_labels = instance['candidate_roles'] #有一些lemma有frame文件，可以取到候选标签
        examples = instance['examples'] #一些例子，用于辅助理解frame文件

        flag=-1
        if candidate_labels == {}:
            flag = 0
            prompt ={"correct": False,
                "reason": "According to lemma'frame, this predicate has no arguments (not judged by LLM)."}
            return prompt, flag # 这个谓词没有相关论元，直接返回是否OK呢，没有经过大模型判断，算是作弊吗？
        else:
            temp_span = sen_span[1].split("-")[-1] # 取最后，不考虑AGM，R-，C-
            if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}: # 标签是核心论元
                if temp_span in candidate_labels.keys():
                    span_mean = candidate_labels[temp_span]
                    prompt = get_prompt_for_corerole(sentence=new_sen_,predicate=predicate,argument=sen_span[0],role=temp_span,role_mean=span_mean)
                    return prompt, flag
                else:
                    flag=1
                    prompt ={"correct": False,
                    "reason": "According to lemma'frame, this predicate has no this label (not judged by LLM)."}
                    return prompt, flag # 这个谓词没有这个标签，直接返回是否OK呢，没有经过大模型判断，算是作弊吗？（其他论文也都这么操作吗，基于lemma的frame进行后处理过滤）
            else: # span不属于核心论元
                span_mean = labels_conll[temp_span]
                prompt = get_prompt_for_otherrole(sentence=new_sen_,predicate=predicate,argument=sen_span[0],role=temp_span,role_mean=span_mean)
                return prompt, flag

        #import pdb;pdb.set_trace()
        #return prompt, flag
    except:
        # 没有frame file的情况
        flag = 2
        temp_span = sen_span[1].split("-")[-1] # 取最后，不考虑R-，C-
        span_mean = Defination[temp_span]
        if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}: # 标签是核心论元
            prompt = get_prompt_for_corerole(sentence=new_sen_,predicate=predicate,argument=sen_span[0],role=temp_span,role_mean=span_mean)
        else:
            prompt = get_prompt_for_otherrole(sentence=new_sen_,predicate=predicate,argument=sen_span[0],role=temp_span,role_mean=span_mean)  
        
        print("-"*50)
        print(f"No frame file: \n\t句子：{sentence}\n\tpredicate: {predicate} \n\tlemma: {lemma}")
        print("-"*50)
        #import pdb;pdb.set_trace()
        return prompt, flag

def get_completion(prompt):
    response = client.chat.completions.create(
                model="gpt-4o-mini",
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
                temperature=0,
                seed=42,
                top_p=1.0,
                n=1,
                max_tokens=200, 
            )
    
    res = response.choices[0].message.content
    
    return res
    
def LLM_prompt(data, path_save):
    # openai.api_timeout = 60

    results=[]

    # 先看下是否之前已经跑了一些结果
    if os.path.exists(path_save):
        results = read_json(path_save)
        final_ins = results[-1]
        for i in range(len(data)):
            if data[i]["index_sen"] == final_ins['index_sen']:
                data = data[i+1:]       
                break  
    print(f'已存在的数据有:{len(results)}')  
    print("-" * 60)
    flag_list = []
    test_num = 0
    for instance in tqdm(data): # 一个句子
        if test_num > 10:
            break
        test_num += 1
        prompt, flag = build_prompt(instance)
        if flag == 0 or flag == 1:
            flag_list.append(flag)
            res = prompt
        else:
            if flag == 2:
                flag_list.append(flag)
            res =  get_completion(prompt)
        instance['response'] = res
        results.append(instance)

        json.dump(results, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
    
    flag_num = Counter(flag_list)
    print("*"*30)
    print(f'frame file selected :{flag_num} \n Sum: {sum(flag_num.values())} \n All span: {len(data)} \n LLM deal: {len(data)-sum(flag_num.values())}')

    print("*"*30)


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
    dataset = ['test'] #dev, 
    source = ["nw",  "bn", "bc" ]# 'bn'
    target = ['tc', 'bn', 'nw', 'bc'] # 'tc',
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                data = read_json(f'forllm_frames_newest/{i}/{i}-{j}-{k}.json')
                path_llmout = f'llmout/{i}/{i}-{j}-{k}-llm.json'
                path_error = f'llmout/{i}/{i}-{j}-{k}-error.json'
                check_data(data, path_error)
                print("数据检查完成！")
                #LLM_prompt(data, path_llmout)
                #print("工作保存完成！")
                #exit()
            
    
