# span-SRL-LLM 代码说明

本目录是一个 SRL (Semantic Role Labeling) 实验工作区，主要研究如何把大模型用于小模型 SRL 结果的校验、过滤、补召回和 gold 数据修正。代码不是封装好的库，很多脚本需要在文件末尾手动切换 `domain`、输入路径、输出路径和模型配置后运行。

## 主要目标

当前代码围绕三件事展开：

1. **LLM 判断候选论元是否正确**：给定句子、谓词、候选 argument span 和角色标签，让 LLM 判断边界是否正确、角色关系是否正确，最终输出 `final_judgement=correct/incorrect`。
2. **融合小模型和大模型**：对 SemiCRF、TreeCRF 或 marginal probability 候选中的 span 做 LLM 过滤或补充，观察 precision、recall、F1 是否提升。
3. **人工修订 gold**：把 LLM、小模型和原始 gold 不一致的样本交给标注者，通过双人标注、IAA、review、merge 得到修订后的 gold。

另外，`prompt_srl/` 里还有一条直接生成式实验线：让 LLM 对某个谓词直接生成所有论元，再和 old gold 或 new gold 计算 PRF。

## 目录结构

```text
ori_code/
├── annotation/             # 主流程：抽样、数据对齐、LLM 判断、人工标注、合并、评估
├── prompt_srl/             # 直接 prompt LLM 生成 SRL 结果并评估
├── preliminary_research/   # 早期探索：候选 span、LLM 判断、投票和错误分析
├── prompt_srl_v2/          # 新版 prompt_srl 数据目录，当前主要放 source_data 和 llm_result
├── log/                    # 早期实验日志
├── summary.md              # 阶段性实验记录和结论
├── github.md               # Git 记录，含敏感信息，不建议继续传播或提交
└── en_core_web_sm-3.7.1-py3-none-any.whl
```

## 数据和下标约定

- 常见 `domain`：`bn`、`tc`、`bc`、`nw`。很多脚本当前默认写死为 `bn` 或 `tc`。
- `prd_idx` 是谓词在句子中的 1-based token index。
- 中间 LLM 数据中的 `span_idx` 多数是 `[start, end)`，即 end-exclusive。
- 人工标注结果中的 `selected_spans` 通常使用 `start` 和 inclusive `end`。
- 评估和 merge 脚本里经常出现 `span_idx[1] - 1`。新增或改脚本前要先确认当前输入的 end 语义。

## 环境依赖

建议从项目根目录开始：

```bash
cd /data/ljwang/span-SRL-LLM/ori_code
```

代码没有统一的 `requirements.txt`。根据现有 import，常用依赖包括：

```text
python, torch, openai, tqdm, numpy, pandas, streamlit, spacy, lemminflect, sympy, matplotlib
```

目录里有 spaCy 英文模型 wheel：

```bash
pip install ./en_core_web_sm-3.7.1-py3-none-any.whl
```

常用外部数据路径：

```text
/data/ljwang/span-SRL-LLM/data/
/data/ljwang/span-SRL-LLM/propbank_frames_main/frame_out/frames_info_3.4.json
/data/ljwang/span-SRL-LLM/treecrf/pred_test/nw/
../../index_sen/nw/
../../prob_distribution/nw/
```

安全注意：`annotation/config.py`、`annotation/7*.py`、`annotation/9*.py`、`prompt_srl/1*.py` 等脚本里存在硬编码 API base_url 和 key。继续使用前建议改成环境变量读取，并轮换已经写进代码的密钥。

## 主流程：annotation/

主流程脚本都在 `annotation/` 下，建议进入该目录运行：

```bash
cd /data/ljwang/span-SRL-LLM/ori_code/annotation
```

### 1. 抽样测试句子

`1.select_test_data.py` 从原始 CoNLL 文件中抽样，过滤无谓词句子和谓词全是 `be` 的句子。

```bash
python 1.select_test_data.py
```

主要输出：

```text
selected_data/test_{domain}_selected_data_{num}.conll
```

运行前检查脚本末尾的 `main("tc", 692)` 或类似设置。

### 2. 解析 gold CoNLL

`2.deal_conll.py` 解析每个谓词的 gold argument span，并去掉 `be` 谓词。

```bash
python 2.deal_conll.py
```

主要输出：

```text
final_data/test_{domain}_goldlabel.conll
```

每条记录包含 `sen`、`prd_word`、`prd_lemma`、`prd.sense`、`pred.idx`、`gold`。

### 3. 对齐小模型预测

```bash
python 3.match_semicrf.py
python 4.match_treecrf.py
```

主要输出：

```text
final_data/{domain}/test_{domain}_goldlabel_semicrflabel.conll
final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel.conll
```

这两步通过句子文本和谓词 index 把 SemiCRF、TreeCRF 结果对齐到 gold 记录上，新增 `semicrf_label` 和 `treecrf_label` 字段。

### 4. 选择高 marginal probability 候选 span

`5.match_high_mp.py` 读取 probability distribution，用阈值筛出候选 span，并去掉已经被 gold、SemiCRF、TreeCRF 覆盖的部分。

```bash
python 5.match_high_mp.py
```

主要输出类似：

```text
final_data/{domain}/test_{domain}_goldlabel_semicrflabel_treecrflabel_highprob_0.1.conll
```

### 5. 构造 LLM 判断输入

`6.forllm.py` 根据 PropBank frame 描述，把 gold、小模型预测或 high_mp 候选改造成 LLM 可判断的单条样本。

```bash
python 6.forllm.py
```

常见输出：

```text
final_data/{domain}/test_{domain}_4llm_core_gold.conll
final_data/{domain}/test_{domain}_4llm_all_gold.conll
final_data/{domain}/test_{domain}_4llm_core_smallmodel.conll
final_data/{domain}/test_{domain}_4llm_all_smallmodel.conll
final_data/{domain}/test_{domain}_4llm_all_smallmodel_repeat.conll
```

核心字段包括 `idx`、`sen`、`prd_word`、`prd_lemma`、`prd_sense`、`prd_idx`、`label`、`span`、`span_idx`、`span_mean`。

### 6. 调用 LLM 判断

核心脚本：

```bash
python 7.llm_judge_gold_o1mini.py
python 7.1llm_judge_smallmodel.py
python 7.2llm_judge_highmp.py
python 9.llm_judge_gold_deepseek.py
python 9.1llm_judge_smallmodel_deepseek.py
```

这些脚本会要求 LLM 输出 JSON，关键字段包括：

```text
argument_extraction_evaluation
role_evaluation
final_judgement
reason_for_* 字段
```

输出通常分两份：

```text
llm/{domain}/test_{domain}_4llm_all_*.conll
llm/{domain}/test_{domain}_4llm_parseright_*.conll
```

第一份保留原始 response，第二份只保留可解析 JSON 的结果。脚本支持根据已有输出断点续跑。

### 7. 后处理 LLM 输出

```bash
python 8.postprocessing.py
python 10.postprocessing_deepseek.py
```

根据 `final_judgement` 拆分：

```text
llm/{domain}/correct_data_{domain}_*.json
llm/{domain}/incorrect_data_{domain}_*.json
```

### 8. 构造人工标注数据

相关脚本：

```basha
python 11.data4annotation.py
python 11.data4annotation_single_gold.py
python 11.data4annotation_single_sm.py
python 11.1data4annotation_repeat.py
python 15.sm_missed_gold_tc.py
```

这些脚本把 LLM 判断结果和原始数据整理成 Streamlit 标注工具需要的 pickle 或 JSON。常见目录是：

```text
anno/
annotated_final/
analysis/
```

### 9. 人工标注和 review UI

单人标注：

```bash
streamlit run 12.streamlit4anno.py --server.address 0.0.0.0 --server.port 8501
```

分歧裁决：

```bash
streamlit run 14.streamlit4review.py --server.address 0.0.0.0 --server.port 8501
```

召回审核：

```bash
streamlit run 24.1review_forrecall_lyh.py --server.address 0.0.0.0 --server.port 8501
streamlit run 24.1review_forrecall_wlj.py --server.address 0.0.0.0 --server.port 8501
```

运行前检查脚本顶部的 `DATA_FILE`、`ANNOTATION_SAVE_FILE`、`OUTPUT_FILE` 或 `SAVE_FILE`。如果端口占用，可以查：

```bash
netstat -tulpn | grep 8501
```

### 10. 标注一致性和分歧抽取

抽取两位标注者的不一致样本：

```bash
python 13.anno_extractionv2.py
```

计算 IAA：

```bash
python 22.newest_annotators_IAA.py
```

也可以指定标注对：

```bash
python 22.newest_annotators_IAA.py \
  --pair name:path/to/a.json:path/to/b.json \
  --details-dir analysis/iaa_details
```

### 11. 合并最终人工修订 gold

BN：

```bash
python 16.merge_final_data.py
```

TC：

```bash
python 16.1merge_final_data_tc.py
```

常见输出：

```text
analysis/test_bn_500_core_final_v4.json
analysis/test_tc_core_final_v2.json
```

这类文件是评估 new gold 的核心输入。

### 12. 评估

大模型判断结果评估：

```bash
python 19.eval_Bigmodel.py
```

小模型在修订 gold 上的评估：

```bash
python 19.1eval_sm.py
```

原始 gold 与修订结果对比：

```bash
python 21.eval_measure_org_repair.py
```

候选 recall review 构建和比较：

```bash
python 23.build_data_forrecallreview_lookup.py
python 24.0build_review.py
python 25.compare_recall.py
```

`23`、`24`、`25` 带 argparse，可以通过 `--help` 查看参数。例如：

```bash
python 24.0build_review.py --sample-size-per-domain 25 --out analysis/span_recall_review_50.json
python 25.compare_recall.py --reviewer-a analysis/span_recall_review_50_lyh.json --reviewer-b analysis/span_recall_review_50_reviewed_wlj.json
```

## prompt_srl/ 直接生成实验

进入目录：

```bash
cd /data/ljwang/span-SRL-LLM/ori_code/prompt_srl
```

主要脚本：

```text
1.0promptSRL-base-gold.py          # 直接让 LLM 为谓词生成所有论元
1.1promptSRL-single-gold.py        # 单角色或变体 prompt
2.0compute_prf_oldgold.py          # 用旧 gold 评估
2.0compute_prf_newgold.py          # 用人工修订 new gold 评估
2.0compute_prf_single_oldgold.py
2.0compute_prf_single_newgold.py
3.0analysis_all_newgold.py         # 分析 LLM 输出与 new gold 的差异
3.1print_llm_extra_generations.py  # 打印 LLM 多生成内容
```

常见数据目录：

```text
old_test_set/
new_test_set/
llm_result/
llm_correct_data/
analysis_data/
log_result/
```

这条线通常基于文本 span 匹配做评估，不同于 `annotation/` 中基于候选 span 的 yes/no QA 判断。

## preliminary_research/ 早期实验

该目录已有 `preliminary_research/README.md`，主要是早期探索代码。常见脚本：

```text
1.obtain_probs_distribution.py       # 获取概率分布
5.select_add_spans.py                # 选择或分析 add 候选 span
7.0build_data4LLM.py                 # 构造 LLM 基础数据
7.1find_frame4labels.py              # 补充 frame 信息，需要 spacy/lemminflect
8.0LLM6_wlj_judge_*.py               # 不同模型的 LLM 判断
9.1.postprocessing-*.py              # 后处理与 PRF
9.4.voting_o1mini_4omini_ds.py       # 多模型投票
```

如果要复现实验主结果，优先看 `annotation/` 和 `prompt_srl/`；`preliminary_research/` 更适合作为早期尝试和错误分析参考。

## 常见输出文件含义

```text
selected_data/                 抽样后的 CoNLL 原文
final_data/                    gold、小模型预测、候选 span 和 LLM 输入数据
llm/                           LLM 原始输出、解析输出、correct/incorrect 拆分
anno/                          Streamlit 标注输入和标注者中间结果
annotated_final/               最终人工标注结果
analysis/                      修订 gold、分歧分析、评估样本、recall review 结果
prompt_srl/llm_result/         直接生成式 prompt SRL 的 LLM 输出
prompt_srl/log_result/         prompt_srl 评估日志
```

## 使用注意事项

1. 运行脚本前先确认当前工作目录。`annotation/` 下的脚本通常假设从 `annotation/` 目录运行。
2. 运行前检查脚本末尾的 `domain`、输入文件和输出文件。当前代码大量使用手动切换注释块的方式管理实验版本。
3. 不要公开或提交 `github.md`、`config.py`、`7*.py`、`9*.py`、`prompt_srl/1*.py` 中的密钥信息。
4. LLM 输出解析依赖模型严格返回 JSON。如果解析失败，要重点检查 `parseright` 文件数量和 postprocessing 输出。
5. 新增评估逻辑时优先复用 `annotation/config_data.py` 中的 `compute_prf`、`build_gold_spans_from_dic`、`build_gold_dict_from_eval` 等函数。
6. `annointerface-old/`、`llm-old-framesenseerror/`、`temp*.py` 多为历史或临时文件，复现实验时不要优先从这些入口开始。

## 推荐阅读顺序

如果只是想理解当前项目，建议按下面顺序看：

1. `summary.md`
2. `annotation/1.select_test_data.py` 到 `annotation/8.postprocessing.py`
3. `annotation/12.streamlit4anno.py`、`13.anno_extractionv2.py`、`16*.merge_final_data*.py`
4. `annotation/19.eval_Bigmodel.py` 和 `annotation/19.1eval_sm.py`
5. `prompt_srl/1*.py`、`prompt_srl/2*.py`

如果要重新跑实验，建议先复制脚本或抽出配置参数，不要直接在已有脚本里继续切换大量注释块，避免覆盖已有结果。
