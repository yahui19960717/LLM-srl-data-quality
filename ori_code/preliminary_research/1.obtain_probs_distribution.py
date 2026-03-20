import os
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第 0 块 GPU
def read(file,device="cuda"):
    data = torch.load(file,map_location=device,mmap=True)
    return data

def obtain_spans_set(data):
    print(f"Batch_size:{len(data.keys())}")
    '''我要获得整个数据集的pred spans/gold spans和候选spans,转换成集合格式'''
    spans_set_dic = {}
    all_gold_spans = []
    all_pred_spans = []
    all_candidate_spans = []
    all_candidate_spans_temp = []
    all_candidate_spans_dic = {}
    # all_mpi = []
    
    sen_idx = 0
    num = 0
    for key in tqdm(data,"processing:"):
        gold_spans = data[key]['gold_spans'] # [(0, 0, 1, 1), (0, 1, 2, 1)]
        pred_spans = data[key]['pred_spans'] # [(0, 1, 1), (1, 2, 1)]
        
        # marginal = data[key]['marginal']
        marginal_probs = data[key]['marginal']
        mask = data[key]['mask']
        chars = data[key]['chars']
        prds = data[key]['prds']
        batch_size, seq_len = chars.shape

        # 用于将logit得分使用softmax得到概率 
        # marginal_probs = torch.softmax(marginal, dim=-1) 
        # 找到谓词的位置,补全没有谓词的句子
        prds_index = torch.zeros(batch_size, dtype=torch.long).to(prds.device) # 将所有的句子谓词的index置于0
        for temp_i in range(batch_size): # 
                if (prds[temp_i]==1).any():
                    index_prd = torch.nonzero(prds[temp_i]== 1, as_tuple=True)[0]
                    prds_index[temp_i]=index_prd
        # 分解每个句子
        for i, gold in enumerate(gold_spans):
            # 对于一句话的gold结果
            pred_id = gold[0][0]
            ''' 遍历每个句子，对于gold 和pred spans加上句子的标识,并获得得分，gold的得分写上1，且去掉null标签的spans'''
            for span in gold:
                # 谓词id如果是0则是表示没有谓词
                if span[3]!=1: # null是1去掉gold_spans中null标签 [句子id，谓词id，论元开始id，论元结束id，得分1] 
                    # all_gold_spans.append((sen_idx, span[0], span[1], span[2], span[3], 1))
                    all_gold_spans.append((sen_idx, span[0], span[1], span[2], span[3], marginal_probs[i][span[1]][span[2]][span[3]].tolist()))
            for span in pred_spans[i]:
                if span[2]!=1 and pred_id!=0: # 去掉pred_spans中的null标签 [句子id，谓词id，论元开始id，论元结束id，得分1]
                    all_pred_spans.append((sen_idx, pred_id, span[0], span[1], span[2], marginal_probs[i][span[0]][span[1]][span[2]].tolist()))

            '''遍历每个句子，获得候选的spans'''
            mpi = marginal_probs[i] # [span_len, span_len, labels] 每个
            # Flatten mask for indexing[spans_len * span_len]
            valid_mask = mask[i].view(-1)  # 薄汗
            # Flatten spans [spans_len * span_len, label_num]
            valid_probs = mpi.view(-1, marginal_probs.size(-1))  
            top_indices = torch.nonzero(valid_mask, as_tuple=True)[0]  # 获得有效spans的idx
            # 先获得所有spans最大值，只有1/label_NUM的数据了，筛选大于阈值的spans
            # [seq_len*seq_len], [seq_len*seq_len], 每个组成的span都有一个最高的标签得分
            scores, indices = valid_probs.max(dim=-1)  # Max score for each span   
            # # 候选的所有spans，去掉了null标签
            candidate_spans_temp = [ # 去掉标签为1（null）的结果
                                    (sen_idx, prds_index[i].item(), idx // seq_len, idx % seq_len+(idx // seq_len), label, scores[idx].item())
                                    for idx, label in zip(top_indices.tolist(), indices[top_indices].tolist())
                                    if label!=1
                                ]
            all_candidate_spans_temp.extend(candidate_spans_temp)
            
            candidate_spans = []
            for idx in top_indices.tolist():
                for label in range(2, mpi.shape[-1]): # 从2开始
                    if valid_probs[idx][label].item()>0:
                        num += 1
                        candidate_spans.append((sen_idx, prds_index[i].item(), idx // seq_len, idx % seq_len+(idx // seq_len), label, valid_probs[idx][label].item()))

            all_candidate_spans.extend(candidate_spans)
            # all_mpi.append((sen_idx, mpi))
            # all_condidate_spans.sort(key=lambda x: x[4], reverse=True)
            sen_idx += 1
        # 获得每个批次的候选spans
        
    spans_set_dic["all_gold_spans"] = all_gold_spans
    spans_set_dic['all_pred_spans'] = all_pred_spans
    spans_set_dic['all_candidate_spans'] = all_candidate_spans_temp
    all_candidate_spans_dic = {temp_key:0 for temp_key in all_candidate_spans}
    # spans_set_dic['all_mpi']  = all_mpi
    print(sen_idx)
    print(f"temp candidate num : {len(all_candidate_spans_temp)}")
    print(f'temp all_candidate_spans: {len(all_candidate_spans)}')
    # for key in all_gold_spans:
    #     if key not in all_candidate_spans.keys():
    #         import pdb;pdb.set_trace()
    print(all(temp_key in all_candidate_spans_dic for temp_key in all_gold_spans)) # 这里出现错误 gold的结果在candidate里面找不到很奇怪
    print(len(spans_set_dic))
    return spans_set_dic
    
def get_all_spans_set(data, file_out):
    file_out = torch.save(data, file_out)

def run(path, file_out):
    data = read(path)
    sets = obtain_spans_set(data) # 用于去掉null标签的spans
    get_all_spans_set(sets, file_out)

if __name__=="__main__":
    

    # 先获得all_spans (pred and gold),去掉null标签
    # 'bc','nw' 'test'
    dataset = [ 'test'] #dev, 
    source = [ "nw"]# 'bn''bn', 'nw', 'bc', 
    target = ['bn']
    for k in dataset:
        for i in source:
            for j in target:
                print(f'{i}-{j}-{k}:')
                path = f"../data-pt/{k}/{i}-{j}-{k}.pt"
                file_out = f"../prob_distribution/{i}/{i}-{j}-{k}-distribution-maximum.pt"
                run(path, file_out)
   

