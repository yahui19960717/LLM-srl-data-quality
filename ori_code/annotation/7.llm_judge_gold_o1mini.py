# 构建prompt来应用LLM
#  wlj老师构造的新版本
import os
import sys
import time
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
from collections import  Counter
from config import Defination,  read_json, labels_conll
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']

random.seed(42)
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",#tcy
    api_key="sk-wUaE2guPPNKLKYTWAhcq7pFNXQMQa6JadrASpFdxsPBbKgkm" # tcy
    # api_key="sk-wUaE2guPPNKLKYTWAhcq7pFNXQMQa6JadrASpFdxsPBbKgkm"
    # sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz
    # sk-uwG3FOUM8NQJswVXoWOLuVYa7OrDoNxd7t7O1it5RLirJiMP
    
)

#核心角色，且知道该谓词frame中有这个角色，重点判断角色是否提取正确
def get_prompt_for_corerole(sentence, predicate, argument, role, role_mean):
    prompt = f""" You are a professional linguistic analysis expert and are now required to perform the semantic role labeling verification task. \
    \n Your task is to evaluate the correctness of the argument and that of the semantic role label for the given sentence, predict, argument and semantic role label. If any errors are found, you must provide a detailed explanation. \
    \n #### Fundamental Definition of SRL:  \
    \n Semantic Role Labeling is a shallow semantic parsing task. Its core objective is to identify the arguments (i.e., participants or states) related to a given predicate and assign them specific semantic roles (e.g., Agent, Patient, Time, Location, etc.). Both the predicate and its arguments are specific words or phrases extracted directly from the sentence. \
    \n **Predicate**: refering to the word or phrase in a clause or sentence that expresses an action, behavior, state, relationship, or attribute. There are four categories of predicates: \
    \n\t 1. verbal predicate: expressing a concrete action or behavior; \
    \n\t 2. nominal predicate: the noun that expresses an event or a relationship can function as a predicate, for example: “The committee announced the appointment of a new director.” -> “appointment” is a noun predicate; \
    \n\t 3. adjectival predicate: expressing an attribute, characteristic, or state of entities and serving as the core meaning in a phrase, clause or sentence; \
    \n\t 4. stative verb/copular verb predicate: expressing the subject's state, attribute, existence, or possession, without involving a concrete action.\
    \n **Core Argument**: refering to the word or phrase that plays a specific role in relation to the predicate, such as agent, patient, theme, instrument, benefactive, attribute, start point or end point, etc. Core arguments are essential components that help complete the meaning of the predicate and provide additional information about the event or state described by the predicate. \
    \n #### Execution Steps and Output Requirements: \
    \n 1. Argument Extraction Verification: \
    \n\t - According to the given predicate and the phrase or clause where it is located, determine whether the given argument is correctly extracted based on syntactic and semantic knowledge. The possible results are: boundary correct, insufficient extraction, over-extraction, boundary incorrect. That is to say, you need to judge whether the boundary of the given argument is correct, based on whether it expresses a complete meaning as the smallest semantic unit. \
    \n\t - If the provided argument span is not extracted correctly, please provide the correct argument span. If the argument you provide is identical to the given argument, you need to revise your previous judgment. \
    \n\t - Output in JSON format with the key fields being argument_extraction_evaluation, reason_for_argment_judgement, and correct_argument. For the field of argument_extraction_evaluation, the candidate value is "yes" or "no". If the provided argument span is correct, the correct_argument field is left empty. \
    \n 2. Semantic Role Correctness Validation: \
    \n\t - Determine whether the argument and the predicate satisfy the specified role relationship. The primary results include: correct relation and incorrect relation. In this task, if the "{argument}" is the "{role_mean}" of the "{predicate}", then they satisfy the relation of "{role}", and the reverse is not true.\
    \n\t - Output in JSON format with the key fields being role_evaluation and reason_for_role_judgement. For the field of role_evaluation, the candidate value is "yes" or "no". \
    \n 3.  Summary of Results: \
    \n\t - According to the results of the argument extraction evaluation and the semantic role correctness validation, provide a summary judgement. If the fields of argument_extraction_evaluation and role_evaluation are both "yes", the final result is correct; if even one value is not "yes", the final result will be incorrect. \
    \n\t - Output in JSON format with the key fields being final_judgement and reason_for_final_judgement. The candidate values for final_judgement are "correct" and "incorrect".  \
    \n #### Input Specification \
    \n We will input a sentence, a predicate, an argument, a role label and the description of this role label. Both the predicate and the argument are words or word fragments from the input sentence. Since there may be multiple identical words in the sentence but at different positions, to better distinguish which word is being referred to, the predicate in the input is marked with underscores (i.e., wrapped with _), and the semantic role field is highlighted in bold (i.e., wrapped with **). \
    \n #### Example \
    \n Input: \
    \n\t Sentence: _Spending_ a lot of time **hunting for bikes** , and nah well . then getting one purchased , and then having to , you know , go get it and things like that . \
    \n\t Predicate: Spending \
    \n\t Argument: hunting for bikes \
    \n\t Role label: ARG2 \
    \n\t Role description: in the clause or sentence where "Spending" is located, "hunting for bikes" represents the "instrument, benefactive, result, caused event, affected entity, topic, content, goal, destination, standard target, comparison target or attribute" of "Spending" \
    \n Output: \
    \n\t {{"argument_extraction_evaluation": "yes", "reason_for_argment_judgement":"The argument \\"hunting for bikes\\" is a contiguous phrase in the sentence and forms a complete semantic unit.", "correct_argument": "", "", "role_evaluation": "yes", "reason_for_role_judgement":"In the clause \\"Spending a lot of time hunting for bikes\\", the argument \\"hunting for bikes\\" specifies the purpose or goal of the predicate \\"Spending\\", which aligns with the role description for ARG2 as it includes goal or content.", "final_judgement": "correct", "reason_for_final_judgement":"Both the argument extraction and the semantic role labeling are correct."}} \
    \n \
    \n Input: \
    \n\t Sentence: **Which** _was_ actually where the major problem came . Exactly . Exactly . yeah . Is that I suddenly had no transportation . \
    \n\t Predicate: was \
    \n\t Argument: Which \
    \n\t Role label: ARG1 \
    \n\t Role description: in the clause or sentence where "was" is located, "Which" represents the "prototypical patient" of "was" \
    \n Output: \n\t {{"argument_extraction_evaluation": "yes", "reason_for_argment_judgement":"The argument \\"Which\\" is correctly extracted as a single word with clear boundaries, serving as a complete semantic unit as a relative pronoun in the sentence.", "correct_argument": "", "", "role_evaluation": "no", "reason_for_role_judgement":"\\"Which\\" functions as the subject or theme, not as a prototypical patient.", "final_judgement": "incorrect", "reason_for_final_judgement":"The argument extraction is correct, but the semantic role label is incorrect because \\"Which\\" does not satisfy the specified role relationship as the prototypical patient of \\"was\\"."}} \
    \n #### Input \
    \n\t Sentence: {sentence} \
    \n\t Predicate: {predicate} \
    \n\t Argument: {argument} \
    \n\t Role label: {role} \
    \n\t Role description: in the clause or sentence where "{predicate}" is located, "{argument}" represents the "{role_mean}" of "{predicate}" \
    \n Please Answer only with a JSON object, without any additional text, explanations, or Markdown formatting. \
    """
    return prompt

#非核心角色，重点判断角色是否提取正确、以及该谓词是否有这个角色
def get_prompt_for_otherrole(sentence, predicate, argument, role, role_mean):
    prompt = f""" You are a professional linguistic analysis expert and are now required to perform the semantic role labeling verification task. \
    \n Your task is to evaluate the correctness of the argument and that of the semantic role label for the given sentence, predict, argument and semantic role label. If any errors are found, you must provide a detailed explanation. and provide a revised solution simultaneously \
    \n #### Fundamental Definition of SRL:  \
    \n Semantic Role Labeling is a shallow semantic parsing task. Its core objective is to identify the arguments (i.e., participants or states) related to a given predicate and assign them specific semantic roles (e.g., Agent, Patient, Time, Location, etc.). Both the predicate and its arguments are specific words or phrases extracted directly from the sentence. \
    \n **Predicate**: refering to the word or phrase in a clause or sentence that expresses an action, behavior, state, relationship, or attribute. There are four categories of predicates: \
    \n\t 1. verbal predicate: expressing a concrete action or behavior; \
    \n\t 2. nominal predicate: the noun that expresses an event or a relationship can function as a predicate, for example: “The committee announced the appointment of a new director.” -> “appointment” is a noun predicate; \
    \n\t 3. adjectival predicate: expressing an attribute, characteristic, or state of entities and serving as the core meaning in a phrase, clause or sentence; \
    \n\t 4. stative verb/copular verb predicate: expressing the subject's state, attribute, existence, or possession, without involving a concrete action.\
    \n **No-core Argument**: refering to as adjuncts (which are used to modify, restrict, or supplement the predicate event, such as condition, manner, direction, cause, purpose or degree etc.) or conjunction words that organize the discourse or connect the context, such as "so", "however", etc. No-core arguments are peripheral components for the predicate. Their omission does not affect the core semantic validity of the event itself, yet they are a crucial part of enriching the semantic description of events. \
    \n #### Execution Steps and Output Requirements: \
    \n 1. Argument Extraction Verification: \
    \n\t - According to the given predicate and the phrase or clause where it is located, determine whether the given argument is correctly extracted based on syntactic and semantic knowledge. The possible results are: boundary correct, insufficient extraction, over-extraction, boundary incorrect. Please note that the given argument may be a modifier of the predicate or peripheral supplementary information, and the judgment of the argument boundary correctness depends on whether the argument semantics are complete. \
    \n\t - If the provided argument span is not extracted correctly, please provide the correct argument span. If the argument you provide is identical to the given argument, you need to revise your previous judgment. \
    \n\t - Output in JSON format with the key fields being argument_extraction_evaluation, reason_for_argument_judgement, and correct_argument. For the field of argument_extraction_evaluation, the candidate value is "yes" or "no". If the provided argument span is correct, the correct_argument field is left empty. \
    \n 2. Semantic Role Correctness Validation: \
    \n\t - Determine whether the argument and the predicate satisfy the specified role relationship. The primary results include: correct relation and incorrect relation. In this task, if the "{argument}" provides the "{role_mean}" about the event or action expressed by the "{predicate}" or be as a modifier of the "{predicate}", then they satisfy the relation of "{role}", and the reverse is not true. Please note that the given argument and predicate may be completely unrelated, or the predicate may not have an argument corresponding to this role relation.\
    \n\t - Output in JSON format with the key fields being role_evaluation and reason_for_role_judgement. For the field of role_evaluation, the candidate value is "yes" or "no".. \
    \n 3.  Summary of Results: \
    \n\t - According to the results of the argument extraction evaluation and the semantic role correctness validation, provide a summary judgement. If the fields of argument_extraction_evaluation and role_evaluation are both "yes", the final result is correct; if even one value is not "yes", the final result will be incorrect. \
    \n\t - Output in JSON format with the key fields being final_judgement and reason_for_final_judgement. The candidate values for final_judgement are "correct" and "incorrect".  \
    \n #### Input Specification \
    \n We will input a sentence, a predicate, an argument, a role label and the description of role label. Both the predicate and the argument are words or word fragments from the input sentence. Since there may be multiple identical words in the sentence but at different positions, to better distinguish which word is being referred to, the predicate in the input is marked with underscores (i.e., wrapped with _), and the semantic role field is highlighted in bold (i.e., wrapped with **). \
    \n #### Example \
    \n Input: \
    \n\t Sentence: Well I guess she did that because someone _suggested_ that to her **because you know the lillies the symbol of the virgin Mary** . \
    \n\t Predicate: suggested \
    \n\t Argument: because you know the lillies the symbol of the virgin Mary \
    \n\t Role label: PRP \
    \n\t Role description: "because you know the lillies the symbol of the virgin Mary" represents "purpose clauses" information of the event expressed by "suggested", or as "purpose clauses"  of "suggested" \
    \n Output: \
    \n\t {{"argument_extraction_evaluation": "yes", "reason_for_argument_judgement":"The argument \\"because you know the lillies the symbol of the virgin Mary\\" is a continuous span in the sentence and forms a complete clause, which is acceptable as an argument unit.", "correct_argument": "", "", "role_evaluation": "no", "reason_for_role_judgement":"The argument \\"because you know the lillies the symbol of the virgin Mary\\" provides a cause or reason for the suggestion, not a purpose. Purpose clauses typically indicate the goal or intention of the agent, whereas this argument explains the external reason (the symbolism of lilies) that motivated the suggestion.", "final_judgement": "incorrect", "reason_for_final_judgement":"The argument extraction is correct, but the semantic role label is incorrect because the argument does not represent a purpose clause of the predicate \\"suggested\\"."}} \
    \n input: \
    \n\t Sentence: %um oh . **So** things are _going_ extremely well . \
    \n\t Predicate: going \
    \n\t Argument: So \
    \n\t Role label: DIS \
    \n\t Role description: "So" represents "conjunction words that organizing the discourse, expressing the speaker\'s attitude, or connecting the context" information of the event expressed by "going", or as "conjunction words that organizing the discourse, expressing the speaker\'s attitude, or connecting the context"  of "going" \
    \n Output: \
    \n\t {{"argument_extraction_evaluation": "yes", "reason_for_argument_judgement":"The argument \\"So\\" is correctly extracted as a single word with clear boundaries, serving as a complete semantic unit as a discourse marker.", "correct_argument": "", "", "role_evaluation": "yes", "reason_for_role_judgement":"In the sentence, \\"So\\" functions as a conjunction that organizes discourse and connects the context to the clause where \\"going\\" is the predicate, aligning with the DIS role description.", "final_judgement": "correct", "reason_for_final_judgement":"Both the argument extraction and the semantic role labeling are correct."}} \
    \n #### Input \
    \n\t Sentence: {sentence} \
    \n\t Predicate: {predicate} \
    \n\t Argument: {argument} \
    \n\t Role label: {role} \
    \n\t Role description: "{argument}" represents "{role_mean}" information of the event expressed by "{predicate}", or as "{role_mean}" of "{predicate}" \
    \n Please Answer only with a JSON object, without any additional text, explanations, or Markdown formatting. \
    """
    return prompt



# flag=0说明谓词没有论元；flag=1说明没有这个标签；flag=2说明没有framefile
def build_prompt(instance):
    sen_id = instance['idx'] #句子id
    sentence = instance['sen'] #句子字符串
    predicate = instance['prd_word'] #给定的谓词字符串
    sen_span = instance['span_idx'] #候选论元信息 (含论元字符串, 角色标签, 论元在句子中的位置，起始位置、终止位置+1)
    label = instance['label']
    lemma = instance['prd_lemma'] #谓词lemma，原型
    prd_id = instance['prd_idx']
    span_mean = instance['span_mean']
    prd_with_ = "_"+predicate+"_" # 谓词加下划线标记，标记谓词，防止多个相同词有歧义
    arg_with_star = "**"+instance['span']+"**" # 论元加粗标记，防止多个相同词有歧义



    #构建新的句子，把谓词用下划线标记，论元用加粗标记
    new_sen = sentence.strip().split().copy()
    new_sen[prd_id-1]=prd_with_
    new_sen = new_sen[0:sen_span[0]]+ [arg_with_star] + new_sen[sen_span[1]:]
    new_sen_ = " ".join(new_sen)
    print(new_sen)
    flag = 0
    try:
        if label in {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}: # 标签是核心论元
            if span_mean != None:
                prompt = get_prompt_for_corerole(sentence=new_sen_,predicate=predicate,argument=instance['span'],role=label,role_mean=span_mean)
                return prompt, flag 
            else:
                if label  in Defination.keys():
                    span_mean = Defination[label]
                    prompt = get_prompt_for_corerole(sentence=new_sen_,predicate=predicate,argument=instance['span'],role=label,role_mean=span_mean)
                    return prompt, flag    
                else:
                    flag=1
                    prompt ={"correct": False,
                    "reason": "According to Defination, this role not exist (not judged by LLM)."}
                    return prompt, flag 
         
    except Exception as e:
        print("数据错误！")
def get_completion(prompt):
    # import pdb;pdb.set_trace()
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

def stas_accuracy(data):
    correct_num = 0
    total_num = len(data) # 所有模型判断的span
    for instance in data:
        final_judgement = instance.get('final_judgement', None)
        error_type = instance.get('error_type', None)
        selected_span = instance.get('selected_span', None)
        org_span = instance.get('org_span', None)
        conflict_span = instance.get('conflict_span', None)
        if final_judgement == 'correct' and (error_type == 'right' or error_type == 'boundary_error' and selected_span[2][0] == conflict_span[2] and selected_span[2][1] == conflict_span[3]):
            correct_num += 1
        elif final_judgement == 'incorrect' and error_type != 'right':
            correct_num += 1
    accuracy = correct_num / total_num if total_num > 0 else 0
    print(f"Total Instances: {total_num}, Correct Instances: {correct_num}, Accuracy: {accuracy:.4f}")
    return accuracy

def LLM_prompt(data, path_save, path_save2):
    # openai.api_timeout = 60

    results = []
    results_simple = []

    # 先看下是否之前已经跑了一些结果
    temp_num = 0
    if os.path.exists(path_save):
        results = read_json(path_save)
        final_ins = results[-1]
        for i in range(len(data)):
            if data[i]["idx"] == final_ins['idx']:
                
                data = data[i+1:]       
                break  
    print(f'已存在的数据有:{len(results)}')  

    if os.path.exists(path_save2):
        results_simple = read_json(path_save2)
        final_ins = results_simple[-1]
        for i in range(len(data)):
            if data[i]["idx"] == final_ins['idx']:
                data = data[i+1:]       
                break  
    print(f'已存在的数据有:{len(results_simple)}')
    print("-" * 60)
    flag_list = []
    for instance in tqdm(data): # 一个句子
        prompt, flag = build_prompt(instance)
        if flag == 1:
            flag_list.append(flag)
            temp_num += 1
            res = prompt
 
        else:
            flag_list.append(flag)
            res =  get_completion(prompt)
            res_data, res_tag = parse_response(instance, res)
            if res_tag == True:
                results_simple.append(res_data)
                # import pdb;pdb.set_trace()
        instance['response'] = res
        results.append(instance)

        json.dump(results, open(path_save, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) 
        json.dump(results_simple, open(path_save2, 'w', encoding="utf-8"), indent=0, ensure_ascii=False) # 只有大模型的结果
    # acc = stas_accuracy(results_simple)
    print("*"*30)
    print(f'\n All span: {len(data)}')
    print(f'错误的个数为： {temp_num}')
    print("*"*30)


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 只使用第 0 块 GPU

    domain = "tc"
    # gold的结果
    # data = read_json( f"final_data/{domain}/test_{domain}_4llm_core_gold.conll")
    # path_llmout = f"llm/{domain}/test_{domain}_4llm_all_gold.conll"
    # path_llmout2 = f"llm/{domain}/test_{domain}_4llm_parseright_gold.conll"
    # LLM_prompt(data, path_llmout, path_llmout2)
    # print("工作保存完成！")

    # small model not recall
    data = read_json( f"final_data/{domain}/test_{domain}_4llm_core_smnotrecall_120.conll")
    path_llmout = f"llm/{domain}/test_{domain}_4llm_all_smnotrecall.conll"
    path_llmout2 = f"llm/{domain}/test_{domain}_4llm_parseright_smnotrecall.conll"
    LLM_prompt(data, path_llmout, path_llmout2)
    print("工作保存完成！")


   


    # ### bn operations ###
    # domain = "bn"
    # # data = read_json( f"final_data/test_{domain}_4llm_core_gold.conll")
    # # path_llmout = f"llm/test_{domain}_4llm_all_gold.conll"
    # # path_llmout2 = f"llm/test_{domain}_4llm_parseright_gold.conll"
    # # LLM_prompt(data, path_llmout, path_llmout2)
    # # print("工作保存完成！")

    # data = read_json(f"final_data/test_{domain}_4llm_core_smnotrecall.conll")
    # path_llmout = f"llm/test_{domain}_4llm_all_smnotrecall.conll"
    # path_llmout2 = f"llm/test_{domain}_4llm_parseright_smnotrecall.conll"
    # LLM_prompt(data, path_llmout, path_llmout2)
    # print("工作保存完成！")
            
    
