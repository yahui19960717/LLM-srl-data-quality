### 数据说明
data:是数据的conll版本，包含了正确的标注
f_index: 是用于将spans和对应的句子对齐，记录了每一批次的句子编号
index_sen: 获取了对应index的句子
forllm_frames : 每个选择出来的span对应着谓词的frame相关内容，比如lemma、例子等
propbank-frames-main:里面存储了frames的相关信息
llmout-old:是之前的输出

### 代码说明
1. 7.0build_data4LLM.py：主要是构建基础信息，句子、谓词、span、错误类型等
2. 7.1find_frame4labels.py：主要是在7.0的基础上添加frame的相关info, 要用到llamafactory的环境
3. 7.2checkdata.py：主要是check数据，王老师写的代码 check boundary



### 工作复盘-与王老师讨论
TODO：
1）模型效果待测：目前使用的是gpt-4o-mini，后面可以针对它判断错误的用其他模型看看效果，是否其他模型更好
2）关于Frame的使用：
    Wang：直接基于golden结果做判断了，都没有走到大模型，算是作弊吗？在实际评测中，可以根据lemma的frame做一下后处理吗，把不符合frame的都去掉？
    Liu：老师这么一说的话，从大模型的角度讲，我感觉是作弊了，这个很少，好像只有39个。实际评测中很少利用frame的，我们能不能说利用先验知识哇？
    Wang：觉得需要做一下对比，若是大模型判断的更好，那其实可以用其对现有标注数据做一下质检，是不是有一些标注错误。现在量比较少，可以先暂缓，但后面还需要需要所有的都经过大模型判断一下。最后大模型在这类上真的不太好，我们再考虑利用先验知识吧
3)现在的谓词，能分出实动词、系动词或助动词、形容词吗
4)升级为大模型看一下上限（用o1 mini试试），看看国内模型是不是更加便宜呢，还有，gpt之前开源了oss，那个模型也还可以的，开源的应该便宜点。将gpt-4o升级为思考模型，成本是如何的，以及是否先升级为大模型看一下上限
5)尝试三个大模型投票,如果一个模型的性能不行的话


2026.1.21
1. 王老师主要是跑了她修改之后prompt的结果，并统计了结果中核心角色和非核心角色的效果：
    Total Instances: 1352, Correct Instances: 935, Accuracy: 0.6916, Previous Accuracy: 0.8010
    Core Role Instances: 775, Correct Core Role Instances: 606, Core Role Accuracy: 0.7819, Previous Core Role Accuracy: 0.8181
    Non-Core Role Instances: 577, Correct Non-Core Role Instances: 329, Non-Core Role Accuracy: 0.5702, Previous Non-Core Role Accuracy: 0.7782 【这里的之前的准确率是指的什么？】
2. 我的工作：
    1） 完成数据的优化和修改，进行了check，目前数据正常：王老师提到的错误，主要是我在数据处理的时候有个if else没有用对，导致selected_span有问题；
    2） 看了王老师的prompt：
        主要是分了核心论元和非核心论元来写的，只要回答是否论元抽取正确性验证，以及角色正确性验证。
        学习到两个异常报错：json.JSONDecodeError和except Exception as e，我之前都是使用了好多if else
    2） 跑生成模型的结果：遇到问题1）思考模型不能用top-p这个参数；2)去掉这个之后发现生成的大部分都是空，查了下资料好像都不太支持，我把其他的参数也去掉了。
    3） 后处理的代码完成

2026.1.20
1. 和王老师讨论，看下王老师如何做的
    1）王老师单独看了一下核心角色和非核心角色的效果，核心角色效果更好一些，非核心角色只有50%。当前还是基于span有问题的统计的。当前统计的时候，没有统计flag=0和flag=1的，就是这里仅统计经过大模型判断的数据
    2）王老师优化了一下Prompt，以及修改了config中角色描述，希望增加模型理解能力，代码都在/data/ljwang/span-SRL-LLM中。同时，我针对gpt-4o-mini错误case，用DeepSeek（https://chat.deepseek.com/）做了测试，都能做对。【想问下王老师是如何测试的？prompt是什么】
    3）王老师用deepseek多一些，不需要代理，还免费
    4）王老师优化了一下Prompt，以及修改了config中角色描述，希望增加模型理解能力
2. 我在改代码的时候遇到的问题：
    1）标签错误，但span交叉算是boundary错误吗？算，已经修改好，加了label的限制： and gold[g]==label:
    2）出现boundary error错误的原因是has_overlap([start, end-1], [g[2], g[3]-1]) 这个边界的判定
    3）增加了dic_span_level['conflict_span']和dic_span_level['gold_label']这两个属性，用来存放错误label的正确label和与span重叠的span。

2026.1.19
1. 王老师和我的工作：
我这边先跑着+评估看看效果（也统计一下不同角色下，LLM判别准确率）
王老师这边尝试改一下Prompt，打算在任务部分加一下当前判别角色的描述，协助模型提升判别能力
2. 王老师给我说数据处理有问题，我看了下，org_span不是记录正确的span，但是我和真实的span对了下，好像也不对，boundary 有问题
重新处理数据，分成frame/no-frame的数据
3. 模型效果待测：目前使用的是gpt-4o-mini，后面可以针对它判断错误的用其他模型看看效果，是否其他模型更好
4. 关于Frame的使用：
    Wang：直接基于golden结果做判断了，都没有走到大模型，算是作弊吗？在实际评测中，可以根据lemma的frame做一下后处理吗，把不符合frame的都去掉？
    Liu：老师这么一说的话，从大模型的角度讲，我感觉是作弊了，这个很少，好像只有39个。实际评测中很少利用frame的，我们能不能说利用先验知识哇？
    Wang：觉得需要做一下对比，若是大模型判断的更好，那其实可以用其对现有标注数据做一下质检，是不是有一些标注错误。现在量比较少，可以先暂缓，但后面还需要需要所有的都经过大模型判断一下。最后大模型在这类上真的不太好，我们再考虑利用先验知识吧

