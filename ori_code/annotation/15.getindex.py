import json
import argparse
from typing import Dict, List, Tuple
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import os
def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def write_json(sentences, path):
    with  open(path, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到: {path}")



# output = "analysis/bn/annotators_analysis_smallmodel163.json"
# output = "analysis/bn/annotators_analysis_gold113.json"
# output = "analysis/bn/annotators_analysis_goldrandom30.json"
# output = "analysis/bn/annotators_analysis_o1right_random30.json"
output =  "analysis/bn/annotators_analysis_botherror_random30.json"

instances = read_json(output)

all_index = [instance['idx'] for instance in instances['records']]
write_json(all_index,"analysis/bn/index_smallmodel_botherror_random30.json" )