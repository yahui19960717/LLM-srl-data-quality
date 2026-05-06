#!/usr/bin/env python3

import os
import sys
import time
from sys import excepthook
from sympy import O
import torch
from tqdm import tqdm
import json
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any

random.seed(42)


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

def get_completion(prompt, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
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
                        #extra_body={"thinking": {"type": "enabled"}}
                    )
            res = response.choices[0].message.content
            return res
        except openai.APITimeoutError as e:
            retries += 1
            print(f"请求超时，正在进行第 {retries} 次重试... ({e})")
            time.sleep(5) # 等待 5 秒后重试
        except Exception as e:
            print(f"发生未知错误: {e}")
            break # 其他错误则直接退出循环
    
    print("重试多次后仍然失败，返回空结果。")
    return "" # 或者可以返回 None 或其他错误标识

# step3: 定义一个函数名称为build_prompt，输入为sentence、prd_word、prd_lemma、prd_sense、roles
# 先构建prompt，然后调用get_completion函数执行prompt，返回执行后的结果
def build_prompt(sentence, prd_word, prd_lemma, prd_sense, roles, single_role):
    prompt = f"""Give a sentence："{sentence}" \n a predicate："{prd_word}" \
    \n all posssible argument roles and their descritions：{roles} \
    \n please label the argument from the sentence that represents "{single_role}" for predicate "{prd_word}". \
    \n If the sentence does not contain the argument that represents "{single_role}", please return "None". \
    \n Provide the answer in JSON format as follows: {{role: argument}} """
    llm_res = get_completion(prompt)
    # import pdb;pdb.set_trace()
    return prompt, llm_res

# step2：写一个函数实现谓词对应角色描述查找；对data中的每一个谓词，根据frames_info_3.4.json查找对应的角色描述
# data中每一条数据含字段sen、prd_word、prd_lemma、prd_sense，以prd_lemma查找frame，再以prd_lemma.prd_sense查找role_set_id，然后返回roles
# 定义一个谓词词典，以sen、prd_word、prd_idx为key，value有prd_sense、roles、Prompt和Prompt执行后的返回结果
def find_role_desc(frames, data, outfile):
    noframe = 0
    pred_dict = {}
    test_num = 0
    print(len(data))
    flag = 0
    not_in_frames = 0
    other_reason = 0
    for key in data:
        #if test_num > 2:
        #    break
        test_num += 1
        sen = key['sen']
        prd_word = key['prd_word']
        prd_lemma = key['prd_lemma'] 
        prd_sense = key['prd_sense']
        prd_idx = key['prd_idx']
        roles = []
        span_mean = key['span_mean']
        if span_mean == None:
            flag += 1
            if prd_lemma in frames:
                senses = frames[prd_lemma]
                for ins in senses:
                    if ins['role_set_id'] == ".".join([prd_lemma, prd_sense]):
                        roles = ins['roles']
                        try:
                            key['span_mean'] = roles[key['label']]
                        except:
                            print(key['label'], prd_lemma, prd_sense)
                            key['span_mean'] = None

    
                            
    for key in data:
        #if test_num > 2:
        #    break
        test_num += 1
        sen = key['sen']
        prd_word = key['prd_word']
        prd_lemma = key['prd_lemma'] 
        prd_sense = key['prd_sense']
        prd_idx = key['prd_idx']
        roles = []
        span_mean = key['span_mean']
        if span_mean == None:
            if prd_lemma not in frames:
                not_in_frames += 1               
            else:
                # print(f"{prd_lemma}.{prd_sense} cannot find")
                # print(frames[prd_lemma])
                # import pdb;pdb.set_trace()
                other_reason+=1
        
    print(flag)
    print(not_in_frames) 
    print(other_reason)
        # import pdb;pdb.set_trace()
        

if __name__ == "__main__":
    domain = "tc" #"bn"
    
    # step1: 读取frames_info_3.4.json和原始gold数据
    # frames = read_json("/data/ljwang/span-SRL-LLM/propbank_frames_main/frame_out/frames_info_3.4.json") 
    frames = read_json("/data/ljwang/span-SRL-LLM/propbank_frames_main/frame_out/frame_info_includephrase.json") 
    original_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold.conll")
    # step2：调用find_role_desc函数，传入frames、original_data、outfile
    outfile = f"llm_result/test_{domain}_4llm_core_gold_single_role.json"
    # outfile = f"llm_result/test_{domain}_4llm_core_gold_single_role_temp_lyh.json"
    find_role_desc(frames, original_data, outfile)
