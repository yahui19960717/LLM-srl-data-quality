# 构建prompt来应用LLM

import os
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
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']

random.seed(42)
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",#tcy
    api_key="sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz" # tcy
)

# flag=0说明谓词没有论元；flag=1说明没有这个标签；flag=2说明没有framefile
def build_prompt(instance):
    sen_id = instance['index_sen']
    sentence = instance['sentences']
    predicate = instance['predicate']
    sen_span = instance['selected_span']
    predict_prob = instance['predict_prob']
    error_type = instance['error_type']
    gold_label = instance['gold_label']
    lemma = instance['lemma']
    org_span = instance["org_span"]
    prd_id = org_span[1]
    prd_with_ = "_"+predicate+"_"
    arg_with_star = "**"+sen_span[0]+"**"

    new_sen = sentence.strip().split().copy()
    new_sen[prd_id-1]=prd_with_
    new_sen = new_sen[0:sen_span[2][0]+1]+ [arg_with_star] + new_sen[sen_span[2][1]:]
    new_sen_ = " ".join(new_sen)

    try:
        candidate_labels = instance['candidate_roles']
        examples = instance['examples']

        flag=-1
        if candidate_labels == {}:
            flag = 0
            prompt ={"correct": False,
                "reason": "This predicate has no arguments."}
            return prompt, flag # 这个谓词没有相关论元
        else:
            temp_span = sen_span[1].split("-")[-1] # 取最后，不考虑R-，C-
            if temp_span in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}: # 标签是核心论元
                if temp_span in candidate_labels.keys():
                    span_mean = candidate_labels[temp_span]
                else:
                    flag=1
                    prompt ={"correct": False,
                    "reason": "This predicate has no this label."}
                    return prompt, flag
            else: # span不属于核心论元
                span_mean = labels_conll[temp_span]
        
            prompt=f"""You are a Semantic Role Labeling Quality Inspector. \
            \n Your task is to evaluate the relationship between a given Predicate and its associated Participant or State within a Sentence. You must verify the existence of the semantic relationship, and the correctness of the assigned Semantic Role. If any errors are found, you must provide a detailed explanation.  \
            \n 1. Fundamental Definition of SRL: Semantic Role Labeling is a shallow semantic parsing task. Its core objective is to identify the participants or states related to a given predicate and assign them specific semantic roles (e.g., Agent, Patient, Time, Location, etc.). Both the predicate and its participant/state are specific words or phrases extracted directly from the sentence. \
            \n 2. Execution Steps and Output Requirements: 
            \n\t Participant or State Existence Verification: 
        \n\t - Determine whether there is a semantic relationship between the given predicate and the specified participant or state. Results include: "related" and "unrelated". \
        \n\t - Output in JSON format, with key fields being whether_semantic_relationship_exists and reason_for_judgment. \ 
        \n\t Semantic Role Correctness Validation: \
        \n\t - Determine whether the labeled participant or state and role relationship are correct based on whether there is a semantic relationship between the predicate and the given participant/state.
        \n\t - According to the given predicate and participant or state, first assess whether the semantic role boundary has been correctly extracted based on syntactic and semantic information. The possible results are: boundary correct, insufficient extraction, over-extraction, boundary incorrect. Then, determine whether the semantic role and the predicate satisfy the specified role relationship. The primary results include: not_have_this_relationship, incorrect_relationship, correct_relationship. \
        \n\t Output in JSON format with the key fields being role_correctness, reason_for_judgment, and correct_result. The reason_for_judgment field stores the boundary_judgment_result and role_relationship_judgment_result. If the role boundary is correct and the role relationship is correct, the correct_result field is left empty. If the predicate does not require this role to be annotated, the correct_result field is filled with Null. If the role boundary is correct but the role relationship is incorrect, the correct_result field is filled with the correct_boundary. If the role relationship is correct but the semantic role boundary is incorrect, the correct_result field is filled with correct_label. For all other cases, the correct_result field is filled with Null. \
        \n 3. Input Specification \
        \n\t We will input a sentence, a predicate, a participant or state, and a semantic role. Both the predicate and the participant or state are words or word fragments from the input sentence. Since there may be multiple identical words in the sentence but at different positions, to better distinguish which word is being referred to, the predicate in the input is marked with underscores (i.e., wrapped with _), and the semantic role field is highlighted in bold (i.e., wrapped with **). \
        \n 4.Example \
        \n\t Input:   \
        \n\t\t Sentence: He **in the park** is _running_.  \
        \n\t\t Predicate: running \
        \n\t\t Participant or State: in the park  \
        \n\t\t Semantic Role: location.
        \n\t Output:  \
        \n\t\t{{"whether_semantic_relationship_exists": "yes","reason_for_judgment": "‘in the park’ provides location information for the action ‘running’, which is one of the contextual participants or states of the event"}}\
        \n\t\t{{"role_correctness":"yes","reason_for_judgment": "semantic role boundary is correct, semantic relationship is correct","correct_result": ""}} \
        \n\t5. Input: \
        \n\t\t Sentence: {new_sen_}\
        \n\t\t Predicate:  {predicate}\
        \n\t\t Participant or State:{sen_span[0]}  \
        \n\t\t Semantic Role: {span_mean} \
        \n\tPlease Answer only with a JSON object.\
        }}"""
        import pdb;pdb.set_trace()
        return prompt, flag
    except:
        # 没有frame file的情况
        flag = 2
        temp_span = sen_span[1].split("-")[-1] # 取最后，不考虑R-，C-
        span_mean = Defination[temp_span]
        prompt =f"""You are a Semantic Role Labeling Quality Inspector. \
            \n Your task is to evaluate the relationship between a given Predicate and its associated Participant or State within a Sentence. You must verify the existence of the semantic relationship, and the correctness of the assigned Semantic Role. If any errors are found, you must provide a detailed explanation.  \
            \n 1. Fundamental Definition of SRL: Semantic Role Labeling is a shallow semantic parsing task. Its core objective is to identify the participants or states related to a given predicate and assign them specific semantic roles (e.g., Agent, Patient, Time, Location, etc.). Both the predicate and its participant/state are specific words or phrases extracted directly from the sentence. \
            \n 2. Execution Steps and Output Requirements: 
            \n\t Participant or State Existence Verification: 
        \n\t - Determine whether there is a semantic relationship between the given predicate and the specified participant or state. Results include: "related" and "unrelated". \
        \n\t - Output in JSON format, with key fields being whether_semantic_relationship_exists and reason_for_judgment. \ 
        \n\t Semantic Role Correctness Validation: \
        \n\t - Determine whether the labeled participant or state and role relationship are correct based on whether there is a semantic relationship between the predicate and the given participant/state.
        \n\t - According to the given predicate and participant or state, first assess whether the semantic role boundary has been correctly extracted based on syntactic and semantic information. The possible results are: boundary correct, insufficient extraction, over-extraction, boundary incorrect. Then, determine whether the semantic role and the predicate satisfy the specified role relationship. The primary results include: not_have_this_relationship, incorrect_relationship, correct_relationship. \
        \n\t Output in JSON format with the key fields being role_correctness, reason_for_judgment, and correct_result. The reason_for_judgment field stores the boundary_judgment_result and role_relationship_judgment_result. If the role boundary is correct and the role relationship is correct, the correct_result field is left empty. If the predicate does not require this role to be annotated, the correct_result field is filled with Null. If the role boundary is correct but the role relationship is incorrect, the correct_result field is filled with the correct_boundary. If the role relationship is correct but the semantic role boundary is incorrect, the correct_result field is filled with correct_label. For all other cases, the correct_result field is filled with Null. \
        \n 3. Input Specification \
        \n\t We will input a sentence, a predicate, a participant or state, and a semantic role. Both the predicate and the participant or state are words or word fragments from the input sentence. Since there may be multiple identical words in the sentence but at different positions, to better distinguish which word is being referred to, the predicate in the input is marked with underscores (i.e., wrapped with _), and the semantic role field is highlighted in bold (i.e., wrapped with **). \
        \n 4.Example \
        \n\t Input:   \
        \n\t\t Sentence: He **in the park** is _running_.  \
        \n\t\t Predicate: running \
        \n\t\t Participant or State: in the park  \
        \n\t\t Semantic Role: location.
        \n\t Output:  \
        \n\t\t{{"whether_semantic_relationship_exists": "yes","reason_for_judgment": "‘in the park’ provides location information for the action ‘running’, which is one of the contextual participants or states of the event"}}\
        \n\t\t{{"role_correctness":"yes","reason_for_judgment": "semantic role boundary is correct, semantic relationship is correct","correct_result": ""}} \
        \n\t5. Input: \
        \n\t\t Sentence: {new_sen_}\
        \n\t\t Predicate:  {predicate}\
        \n\t\t Participant or State:{sen_span[0]}  \
        \n\t\t Semantic Role: {span_mean} \
        \n\tPlease Answer only with a JSON object.\
        }}"""
        
        print("-"*50)
        print(f"No frame file: \n\t句子：{sentence}\n\tpredicate: {predicate} \n\tlemma: {lemma}")
        print("-"*50)
        import pdb;pdb.set_trace()
        return prompt, flag
def get_completion(prompt):
    response = client.chat.completions.create(
                model="o1-mini",
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
    for instance in tqdm(data): # 一个句子
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
                data = read_json(f'forllm_frames/{i}/{i}-{j}-{k}.json')
                path_llmout = f'llmout/llmout_wlj/{i}/{i}-{j}-{k}.json'
                LLM_prompt(data, path_llmout)
                print("工作保存完成！")
                exit()
            
    