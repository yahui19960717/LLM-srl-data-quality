import torch
import json
import openai, json
import pickle
from openai import OpenAI
conll12_label = ['<pad>', 'O', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA', 'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP', 'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG', 'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP', 'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-MOD', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP', 'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-CAU', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR', 'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRP', 'R-ARGM-TMP', 'R-ARGM-PRD',]
conll05_label = ['<pad>', 'O', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'AA', 'AM', 'AM-ADV', 'AM-CAU', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PNC', 'AM-PRD', 'AM-REC', 'AM-TMP', 'C-A0', 'C-A1', 'C-A2', 'C-A3', 'C-A4', 'C-A5', 'C-AM-ADV', 'C-AM-CAU', 'C-AM-DIR', 'C-AM-DIS', 'C-AM-EXT', 'C-AM-LOC', 'C-AM-MNR', 'C-AM-NEG', 'C-AM-PNC', 'C-AM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-A3', 'R-A4', 'R-AA', 'R-AM-ADV', 'R-AM-CAU', 'R-AM-DIR', 'R-AM-EXT', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-PNC', 'R-AM-TMP', 'V']
ARG = {
    "ARG0": "prototypical agent: the deliberate instigator or doer of the action/event.",
    "ARG1": "prototypical patient: the entity directly affected or created by the action; the ‘thing acted upon’.",
    "ARG2": "instrument, benefactive, or attribute: the means used to perform the action, the recipient/beneficiary, or the assigned attribute value.",
    "ARG3": "starting point, benefactive, or attribute: the origin, source location, or (for some verbs) the beneficiary/attribute.",
    "ARG4": "ending point: the destination, goal, or final location of the action or motion.",
    "ARG5": "",
    "ARGA": "represents a secondary agent, an argument that plays an agentive role but is not the primary agent."
}
ARGM = {
'ARGM-ADJ': "used to label adjectival modifiers.",
'ARGM-ADV': "used for adverbial modifiers that modify the verb's event structure",
'ARGM-CAU': "similar to purpose clauses, these indicate the reason for an action", 
'ARGM-COM': "indicate who an action was done with.",
'ARGM-DIR': "show motion along some path.", 
'ARGM-DIS': "it connect a sentence to a preceding sentence",
'ARGM-DSP': "it has a speaker/thinker argument (ARG0) and the utterance/thought (ARG1).",
'ARGM-EXT': "indicate the amount of change occurring from an action",
'ARGM-GOL': "This tag is for the goal of the action of the verb. ",
'ARGM-LOC': "indicate where some action takes place.", 
'ARGM-LVB': "used to label the light verb only in the noun pass of light verb annotation", 
'ARGM-MNR': "Manner adverbs specify how an action is performed.", 
'ARGM-MOD': "Modals are: will, may, can, must, shall, might, should, could, and would.", 
'ARGM-NEG': "This tag is used for elements such as not, n’t, never, no longer, and other markers of negative sentences", 
'ARGM-PNC': "", 
'ARGM-PRD': "It shows that an adjunct of a predicate is in itself capable of carrying some predicate structure", 
'ARGM-PRP': "Purpose clauses are used to show the motivation for some action", 
'ARGM-PRR': "", 
'ARGM-PRX': "", 
'ARGM-REC': "These include reflexives and reciprocals such as himself, itself, themselves, each other, or own, which refer back to one of the other arguments.", 
'ARGM-TMP': "It shows when an action took place", 
}

# Defination = {
#     "ARG0": "prototypical agent: the deliberate instigator or doer of the action/event.",
#     "ARG1": "prototypical patient: the entity directly affected or created by the action; the ‘thing acted upon’.",
#     "ARG2": "instrument, benefactive, or attribute: the means used to perform the action, the recipient/beneficiary, or the assigned attribute value.",
#     "ARG3": "starting point, benefactive, or attribute: the origin, source location, or (for some verbs) the beneficiary/attribute.",
#     "ARG4": "ending point: the destination, goal, or final location of the action or motion.",
#     "ARG5": "",
#     "ARGA": "represents a secondary agent, an argument that plays an agentive role but is not the primary agent.",
#     'ADJ': "used to label adjectival modifiers.",
#     'ADV': "used for adverbial modifiers that modify the verb's event structure",
#     'CAU': "similar to purpose clauses, these indicate the reason for an action", 
#     'COM': "indicate who an action was done with.",
#     'DIR': "show motion along some path.", 
#     'DIS': "it connect a sentence to a preceding sentence",
#     'DSP': "it has a speaker/thinker argument (ARG0) and the utterance/thought (ARG1).",
#     'EXT': "indicate the amount of change occurring from an action",
#     'GOL': "This tag is for the goal of the action of the verb. ",
#     'LOC': "indicate where some action takes place.", 
#     'LVB': "used to label the light verb only in the noun pass of light verb annotation", 
#     'MNR': "Manner adverbs specify how an action is performed.", 
#     'MOD': "used to label modal verbs like will, may, can, must, shall, might, should, could, and would.", 
#     'NEG': "This tag is used for elements such as not, n’t, never, no longer, and other markers of negative sentences", 
#     'PNC': "used to indicate purpose or condition.", 
#     'PRD': "It shows that an adjunct of a predicate is in itself capable of carrying some predicate structure", 
#     'PRP': "purpose clauses are used to show the motivation for some action.", 
#     'PRR': "purpose", 
#     'PRX': "used to indicate additional information or extension related to a previously mentioned entity, providing further clarification or detail.", 
#     'REC': "These include reflexives and reciprocals such as himself, itself, themselves, each other, or own, which refer back to one of the other arguments.", 
#     'TMP': "it shows when an action took place. ", 
#     'R': "R representing arguments realized in other parts of the sentence. ",
#     'C': "C indicates a continuation. ",
# }


Defination = {
    "ARG0": "prototypical agent",
    "ARG1": "prototypical patient",
    "ARG2": "instrument, benefactive, or attribute",
    "ARG3": "starting point, benefactive, or attribute",
    "ARG4": "ending point",
    "ARG5": "agent",
    "ARGA": "a secondary agent",
    'ADJ': "adjectival modifiers.",
    'ADV': "adverbial modifiers",
    'CAU': "purpose clauses", 
    'COM': "comitative",
    'DIR': "direction", 
    'DIS': "discourse",
    'DSP': "direct speech",
    'EXT': "extent",
    'GOL': "goal",
    'LOC': "location", 
    'LVB': "light verb", 
    'MNR': "manner", 
    'MOD': "general modification", 
    'NEG': "negation", 
    'PNC': "purpose no cause (no longer used)", 
    'PRD': "secondary predication", 
    'PRP': "purpose", 
    'PRR': "purpose", 
    'PRX': "prepositional phrase extension", 
    'REC': "recipricol (eg herself, etc)", 
    'TMP': "temporal", 
    'R': "R representing arguments realized in other parts of the sentence. ",
    'C': "C indicates a continuation. ",
}
labels_conll = {
    'EXT': 'extent', 
    'LOC': 'location', 
    'DIR': 'direction', 
    'NEG': 'negation', 
    'MOD': 'general modification', 
    'ADV': 'adverbial modification', 
    'MNR': 'manner', 
    'PRD': 'secondary predication', 
    'REC': 'recipricol (eg herself, etc)', 
    'TMP': 'temporal', 
    'PRP': 'purpose', 
    'PNC': 'purpose no cause (no longer used)', 
    'CAU': 'cause', 
    'CXN': 'constructional pattern (adjectival comparative marker)', 
    'ADJ': 'adjectival (nouns only)', 
    'COM': 'comitative', 
    'DIS': 'discourse', 
    'DSP': 'direct speech', 
    'GOL': 'goal', 
    'PAG': 'prototypical agent (function tag for arg1)', 
    'PPT': 'prototypical patient (function tag for arg1)', 
    'RCL': 'relative clause (no longer used)', 
    'SLC': 'selectional constraint link', 
    'VSP': 'verb specific (function tag for numbered arguments)', 
    'LVB': 'light verb (for nouns only)',
    'PRX': 'prepositional phrase extension',
    'COM': "comitative",
    'ARGA': 'represents a secondary agent, an argument that plays an agentive role but is not the primary agent.'}

def read(file,device="cuda"):
    data = torch.load(file,map_location=device,mmap=True, weights_only=True)
    return data

# 写json文件 
def write_json(data, file):
    with open(file, 'w') as json_file:
            json.dump(data, json_file, indent=0)
# 读json文件 index
def read_json(file):
  with open(file, "r", encoding="utf-8") as f:
      data = json.load(f)
  return data 

client = OpenAI(
    # base_url="https://api.gptapi.us/v1/chat/completions",
    # base_url="https://m.gptapi.us/v1",
    # api_key="sk-PFxesFI2yhSscA1mBaEd21D58d8b45Ae94AdCc1f51706a03"
    # api_key="sk-ZVJZkkjAVYZWKdnl9bD40dC3D79a4a19BbC2B817Db475095" #0717
    # base_url="https://api.nuwaapi.com/v1/chat/completions",
    # api_key="sk-uTOLK2xd8Sdzjh89PodKnsldYpTaUBihKNWAE68JlfFJtpyP" # haozhe
    base_url="https://api2.aigcbest.top/v1",#tcy
    api_key="sk-rHRiBjpZe6ssAP67Su0raBRKnJ0K9jHys5nywBnTFjcmW1U1" # tcy
    
    
)

def get_completion(prompt):
    response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content":"You are a linguistics expert. "
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

def read_pickle(file):
    # 从文件读取
    with open(file, 'rb') as f:  # 注意：要用二进制模式 'rb'
        data = pickle.load(f)
    print(len(data))
    return data
