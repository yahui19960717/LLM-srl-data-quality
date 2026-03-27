#!/usr/bin/env python3

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
from collections import defaultdict
from typing import Dict, List, Tuple, Any

random.seed(42)
client = OpenAI(
    base_url="https://api2.aigcbest.top/v1",#tcy
    # api_key="sk-uwG3FOUM8NQJswVXoWOLuVYa7OrDoNxd7t7O1it5RLirJiMP" # tcy
    #api_key="sk-7k63FEwvaVFJnUD91JNozH9LUDHoeiH5MCb8Hjt9a5I9UY04"
    api_key="sk-7k63FEwvaVFJnUD91JNozH9LUDHoeiH5MCb8Hjt9a5I9UY04"
    # sk-lQZso15BZziwxSs6mOjpHPHl6AX832tMh4g8FZwa444vKOSz
    
)

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")

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

# step3: 定义一个函数名称为build_prompt，输入为sentence、prd_word、prd_lemma、prd_sense、roles
# 先构建prompt，然后调用get_completion函数执行prompt，返回执行后的结果
def build_prompt(sentence, prd_word, prd_lemma, prd_sense, roles):
    prompt = f"""Give a sentence："{sentence}", a predicate："{prd_word}", all posssible argument roles and their descritions：{roles}, please label all the possible argument about predicate "{prd_word}" of the sentence.\n Provide the answer in JSON format as follows: {{"comment":{{role: argument, role: argument, ...}}}} """
    llm_res = get_completion(prompt)
    return prompt, llm_res

# step2：写一个函数实现谓词对应角色描述查找；对data中的每一个谓词，根据frames_info_3.4.json查找对应的角色描述
# data中每一条数据含字段sen、prd_word、prd_lemma、prd_sense，以prd_lemma查找frame，再以prd_lemma.prd_sense查找role_set_id，然后返回roles
# 定义一个谓词词典，以sen、prd_word、prd_idx为key，value有prd_sense、roles、Prompt和Prompt执行后的返回结果
def find_role_desc(frames, data, outfile):
    noframe = 0
    pred_dict = {}
    test_num = 0
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
        if prd_lemma in frames:
            senses = frames[prd_lemma]
            for ins in senses:
                if ins['role_set_id'] == ".".join([prd_lemma, prd_sense]):
                    roles = ins['roles']
                    #print(roles)
                    pred_key = "\t".join([sen, prd_word, str(prd_idx)])
                    if pred_key not in pred_dict:
                        prompt, llm_res = build_prompt(sen, prd_word, prd_lemma, prd_sense, roles)
                        #print(prompt)
                        #print(llm_res)
                        pred_dict[pred_key] = {"prd_sense": prd_sense, "roles": roles, "Prompt": prompt, "Prompt_Result": llm_res}
        else:
            noframe += 1
            print(f"未找到{prd_lemma}的角色描述, sen: {sen}, prd_word: {prd_word}, prd_idx: {prd_idx}")
    # 将pred_dict中内容写到json文件
    write_json(pred_dict, outfile)

if __name__ == "__main__":
    domain = "tc" # or bn
    
    # step1: 读取frames_info_3.4.json和原始gold数据
    frames = read_json("/data/ljwang/span-SRL-LLM/propbank_frames_main/frame_out/frames_info_3.4.json") 
    original_data = read_json(f"/data/ljwang/span-SRL-LLM/ori_code/annotation/final_data/{domain}/test_{domain}_4llm_core_gold.conll")
    # step2：调用find_role_desc函数，传入frames、original_data、outfile
    outfile = f"llm_result/test_{domain}_4llm_core_gold_role.json"
    find_role_desc(frames, original_data, outfile)
