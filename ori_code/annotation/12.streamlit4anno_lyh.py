"""
标注界面
可以选择spans，也可以自定义
"""
import os
import streamlit as st
import pandas as pd
import json
import pickle
from collections import defaultdict
from datetime import datetime
import re
# import socket
# 数据文件路径
# DATA_FILE = "anno/bn_annotation_gold_o1right_random.pkl"
# DATA_FILE = "anno/bn_annotation_smallmodel.pkl"
# DATA_FILE = "anno/bn_annotation_single_gold.pkl"
# DATA_FILE = "anno/bn_annotation_single_gold_random.pkl"
# DATA_FILE = "anno/bn_annotation_smallmodel_botherror_random.pkl"
DATA_FILE = "anno/bn_annotation_single_gold_o1wrongdsright_93.pkl"
ANNOTATION_SAVE_FILE = "anno/annotations_gold_o1wrongdsright_lyh93.json"
def read_pickle(file):
    """从文件读取pickle数据"""
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(len(data))
    return data

# # 获取服务器IP
# hostname = socket.gethostname()
# local_ip = socket.gethostbyname(hostname)

# 设置页面配置
st.set_page_config(
    page_title="语义角色标注工具",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 紧凑布局优化
st.markdown("""
<style>
    /* 全局紧凑布局 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }

    /* 主标题样式 - 缩小 */
    .main-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.3rem 0;
        margin-bottom: 0.5rem;
    }

    /* 句子显示区域 - 紧凑 */
    .sentence-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 1.05rem;
        line-height: 1.6;
        margin: 0.4rem 0;
        border-left: 3px solid #667eea;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* 预览区域 - 紧凑 */
    .preview-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 1rem;
        line-height: 1.5;
        margin: 0.4rem 0;
        border-left: 3px solid #ff6b6b;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* 高亮样式优化 */
    .predicate-highlight {
        color: #d63031;
        font-weight: 700;
        background: linear-gradient(120deg, #ffe6e6 0%, #ffd7d7 100%);
        padding: 2px 5px;
        border-radius: 3px;
        box-shadow: 0 1px 2px rgba(214, 48, 49, 0.2);
    }

    .argument-highlight {
        background: linear-gradient(120deg, #fff3cd 0%, #ffe69c 100%);
        font-weight: 600;
        padding: 2px 5px;
        border-radius: 3px;
        box-shadow: 0 1px 2px rgba(255, 193, 7, 0.3);
    }

    /* 按钮样式 - 紧凑 */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        padding: 0.3rem 0.8rem !important;
        font-size: 0.9rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }

    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* 进度条样式 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* 信息卡片 - 紧凑 */
    .info-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        margin: 0.3rem 0;
        border-left: 2px solid #00acc1;
        font-size: 0.9rem;
    }

    /* 标注状态徽章 - 紧凑 */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .status-annotated {
        background: linear-gradient(120deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 1px solid #c3e6cb;
    }

    .status-pending {
        background: linear-gradient(120deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 1px solid #ffeaa7;
    }

    /* 分隔线 - 紧凑 */
    hr {
        margin: 0.8rem 0 !important;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }

    /* 下标样式 */
    sub {
        color: #6c757d;
        font-size: 0.65em;
        font-weight: 500;
        margin-left: 2px;
    }

    /* Streamlit 原生组件间距调整 */
    .stMarkdown {
        margin-bottom: 0.3rem !important;
    }

    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }

    .stRadio > label {
        font-size: 0.9rem !important;
        margin-bottom: 0.2rem !important;
    }

    .stTextArea > label, .stNumberInput > label {
        font-size: 0.9rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* 表单间距 */
    .stForm {
        border: none !important;
        padding: 0.5rem 0 !important;
    }

    /* 成功/错误/警告/信息提示 - 紧凑 */
    .stAlert {
        padding: 0.4rem 0.8rem !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)



st.markdown('<h1 class="main-title">🏷️ 语义角色标注工具</h1>', unsafe_allow_html=True)

# 初始化session state
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'data' not in st.session_state:
    # 加载数据
    if os.path.exists(DATA_FILE):
        try:
            st.session_state.data = read_pickle(DATA_FILE)
            st.success(f"✅ 成功加载 {len(st.session_state.data)} 条数据")
        except Exception as e:
            st.error(f"❌ 加载数据失败: {e}")
            st.session_state.data = []
    else:
        st.error(f"❌ 数据文件不存在: {DATA_FILE}")
        st.session_state.data = []

def save_annotations():
    """保存标注结果"""
    try:
        with open(ANNOTATION_SAVE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'annotations': st.session_state.annotations,
                'last_index': st.session_state.current_idx
            }, f, ensure_ascii=False, indent=2)
        st.success(f"💾 标注结果已保存")
    except Exception as e:
        st.error(f"❌ 保存失败: {e}")

def load_annotations():
    """加载之前的标注结果"""
    if os.path.exists(ANNOTATION_SAVE_FILE):
        try:
            with open(ANNOTATION_SAVE_FILE, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                st.session_state.annotations = saved_data.get('annotations', [])
                st.session_state.current_idx = saved_data.get('last_index', 0)
            st.success(f"📂 已加载标注进度，当前在第 {st.session_state.current_idx + 1} 条")
            return True
        except Exception as e:
            st.error(f"❌ 加载标注文件失败: {e}")
    return False

def get_span_text(sentence, start, end):
    """从句子中提取指定范围的文本"""
    tokens = sentence.split()
    if start <= end and 0 <= start < len(tokens) and 0 < end <= len(tokens):
        return ' '.join(tokens[start:end])
    return ""

def display_sentence_with_highlights(sentence, prd_idx, selected_span=None, show_index=False):
    """显示带有高亮的句子（prd_idx为0-based索引）"""
    tokens = sentence.split()
    html_parts = []

    for i, token in enumerate(tokens):
        index_tag = f'<sub>{i}</sub>' if show_index else ''
        if i == prd_idx:
            html_parts.append(f'<span class="predicate-highlight">{token}{index_tag}</span>')
        elif selected_span and selected_span[0] <= i <= selected_span[1]:
            html_parts.append(f'<span class="argument-highlight">{token}{index_tag}</span>')
        else:
            html_parts.append(f'{token}{index_tag}')

    return ' '.join(html_parts)

def get_annotation_status(current_idx):
    """获取当前句子的标注状态"""
    for ann in st.session_state.annotations:
        if ann['idx'] == current_idx:
            return ann
    return None

def slide_bar(st):
    """侧边栏：进度信息和跳转"""
    with st.sidebar:
        st.markdown("**📊 进度**")

        # 加载/保存按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 加载", use_container_width=True, type="secondary"):
                load_annotations()
                st.rerun()
        with col2:
            if st.button("💾 保存", use_container_width=True, type="primary"):
                save_annotations()

        # 显示标注统计
        total_sentences = len(st.session_state.data)
        annotated_indices = {ann['idx'] for ann in st.session_state.annotations}
        annotated_count = len(annotated_indices)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("总数", total_sentences, label_visibility="visible")
        with col2:
            st.metric("已标注", f"{annotated_count}", label_visibility="visible")

        # 显示标注进度条
        progress = annotated_count / total_sentences if total_sentences > 0 else 0
        st.progress(progress, text=f"{progress*100:.1f}%")

        # 跳转到指定序号
        st.markdown("---")
        st.markdown("**🔍 跳转**")
        jump_idx = st.number_input(
            f"序号 (1-{total_sentences})",
            min_value=1,
            max_value=total_sentences,
            value=st.session_state.current_idx + 1,
            key="jump_input",
            label_visibility="collapsed"
        )
        if st.button("🎯 跳转", use_container_width=True, type="primary"):
            st.session_state.current_idx = jump_idx - 1
            st.rerun()

        # 快捷跳转按钮
        st.markdown("---")
        st.markdown("**⚡ 快捷**")
        if st.button("⏭️ 下一未标注", use_container_width=True):
            for i in range(st.session_state.current_idx + 1, len(st.session_state.data)):
                if st.session_state.data[i]['idx'] not in annotated_indices:
                    st.session_state.current_idx = i
                    st.rerun()
                    break
            else:
                st.warning("⚠️ 无未标注")

        if st.button("⏮️ 首个未标注", use_container_width=True):
            for i in range(len(st.session_state.data)):
                if st.session_state.data[i]['idx'] not in annotated_indices:
                    st.session_state.current_idx = i
                    st.rerun()
                    break
            else:
                st.info("🎉 全部完成")

def main_top(st, current_data, annotation_status):
    """顶部：显示当前句子信息和标注状态"""
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.markdown(f"**📝 句子 {st.session_state.current_idx + 1} / {len(st.session_state.data)}**")
    with col_header2:
        if annotation_status:
            st.markdown('<span class="status-badge status-annotated">✅ 已标注</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-pending">⏳ 未标注</span>', unsafe_allow_html=True)

    # 显示基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="info-card"><b>序号:</b> {current_data["idx"]}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="info-card"><b>谓词:</b> {current_data["prd_word"]}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="info-card"><b>标签:</b> {current_data["label"]}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="info-card"><b>论元意思:</b> {current_data["span_mean"]}</div>', unsafe_allow_html=True)

def grammar(current_data, st):
    """语法检查部分"""
    st.markdown("**🔍 检查句子**")

    current_idx = current_data['idx']

    # 获取已保存的语法检查结果
    saved_grammar = None
    saved_error_desc = ""
    for ann in st.session_state.annotations:
        if ann['idx'] == current_idx:
            saved_grammar = ann.get('grammar_status')
            saved_error_desc = ann.get('grammar_error_desc', '')
            break

    col1, col2 = st.columns([1, 2])

    with col1:
        grammar_status = st.radio(
            "句子是否有语法错误？",
            options=["✅ 没有语法错误", "❌ 有语法错误"],
            index=0 if saved_grammar is None or saved_grammar == "没有语法错误" else 1,
            key=f"grammar_radio_{current_idx}",
            label_visibility="collapsed"
        )

    with col2:
        if grammar_status == "❌ 有语法错误":
            error_description = st.text_area(
                "请描述语法错误：",
                value=saved_error_desc,
                key=f"grammar_desc_{current_idx}",
                placeholder="例如：缺少主语、搭配不当、语序错误等",
                height=60,
                label_visibility="collapsed"
            )
            if error_description and error_description.strip():
                st.success("✓ 已标记为句子错误", icon="✅")
                grammar_check_passed = True
            else:
                st.error("❌ 请填写错误描述")
                grammar_check_passed = False
        else:
            error_description = ""
            st.success("✓ 句子无错误", icon="✅")
            grammar_check_passed = True

    # 处理返回值中的emoji
    grammar_status_clean = grammar_status.replace("✅ ", "").replace("❌ ", "")
    return grammar_status_clean, error_description, grammar_check_passed

def main():
    # 1. 检查数据是否加载成功
    if not st.session_state.data:
        st.error("❌ 没有加载到数据，请检查文件路径")
        return

    # 2. 侧边栏
    slide_bar(st)

    # 3. 主内容区
    current_data = st.session_state.data[st.session_state.current_idx]
    annotation_status = get_annotation_status(current_data['idx'])

    main_top(st, current_data, annotation_status)

    # 3.1. 显示完整句子
    st.markdown("**📄 完整句子**")
    temp_sen = " ".join(current_data['sen'].split())
    st.markdown(current_data['idx'],  unsafe_allow_html=True)
    st.markdown(temp_sen,  unsafe_allow_html=True)
    prd_idx_0based = current_data['prd_idx'] - 1
    html_sentence = display_sentence_with_highlights(
        current_data['sen'],
        prd_idx_0based,
        selected_span=None,
        show_index=True
    )
    
    st.markdown(f'<div class="sentence-box">{html_sentence}</div>', unsafe_allow_html=True)

    # 3.2. 语法检查
    if 'grammar_check' not in st.session_state:
        st.session_state.grammar_check = {}
    grammar_status, error_description, grammar_check_passed = grammar(current_data, st)

    # 3.3 标注论元
    st.markdown("**🎯 标注论元**")

    # 显示已标注的span（如果有）- 兼容 selected_spans（新）和 selected_span（旧）
    saved_spans = None
    if annotation_status:
        if 'selected_spans' in annotation_status:
            saved_spans = annotation_status['selected_spans']
        elif 'selected_span' in annotation_status:
            saved_spans = [annotation_status['selected_span']]

    if saved_spans:
        st.caption(f"当前已标注的论元（{len(saved_spans)} 个候选）:")
        for sp in saved_spans:
            selected_span = (sp['start'], sp['end'])
            html_sentence = display_sentence_with_highlights(
                current_data['sen'],
                prd_idx_0based,
                selected_span
            )
            st.markdown(f'<div class="preview-box">{html_sentence}</div>', unsafe_allow_html=True)

    # 构建选项数据
    options = current_data['options']
    option_mappings = []

    if isinstance(options, (defaultdict, dict)):
        for (start, end), models in options.items():
            span_text = get_span_text(current_data['sen'], start, end)
            if span_text:
                tokens = current_data['sen'].split()
                highlighted_tokens = []
                for i, token in enumerate(tokens):
                    if start <= i < end:
                        highlighted_tokens.append(f'<span class="argument-highlight">{token}</span>')
                    elif i == prd_idx_0based:
                        highlighted_tokens.append(f'<span class="predicate-highlight">{token}</span>')
                    else:
                        highlighted_tokens.append(token)
                highlighted_sentence = ' '.join(highlighted_tokens)
                option_mappings.append({
                    'start': start,
                    'end': end-1,
                    'text': span_text,
                    'models': models,
                    'highlighted_sentence': highlighted_sentence
                })

    # 构建选项标签
    option_labels = [f"选项 {i+1}: '{d['text']}'" for i, d in enumerate(option_mappings)]
    option_labels.append("🚫 无论元")
    option_labels.append("✏️ 自定义标注")

    # 设置默认选中的选项
    default_index = 0
    if annotation_status:
        saved_spans = annotation_status.get('selected_spans')
        if saved_spans is None and 'selected_span' in annotation_status:
            saved_spans = [annotation_status['selected_span']]
        if saved_spans is not None and len(saved_spans) == 0:
            # 无论元
            default_index = len(option_mappings)
        elif saved_spans:
            first_span = saved_spans[0]
            models = first_span.get('models', [])
            if models == ['自定义']:
                default_index = len(option_labels) - 1
            else:
                for i, mapping in enumerate(option_mappings):
                    if mapping['start'] == first_span['start']:
                        default_index = i
                        break

    # radio 放在 form 外面，实时预览
    select_option = st.radio(
        "选择标注选项:",
        range(len(option_labels)),
        format_func=lambda x: option_labels[x],
        index=default_index,
        key=f"option_radio_{st.session_state.current_idx}",
        label_visibility="collapsed"
    )

    # 实时预览选中的选项
    if select_option < len(option_mappings):
        selected_mapping = option_mappings[select_option]
        st.caption("预览效果:")
        st.markdown(f'<div class="preview-box">{selected_mapping["highlighted_sentence"]}</div>', unsafe_allow_html=True)
        st.caption(f"来源: {', '.join(selected_mapping['models'])}  |  位置: [{selected_mapping['start']}:{selected_mapping['end']}]")
    

        # 补充候选区域
        extra_key = f"extra_spans_{st.session_state.current_idx}"
        add_extra = st.checkbox("➕ 补充其他候选边界", key=f"add_extra_{st.session_state.current_idx}")

        if add_extra:
            if extra_key not in st.session_state:
                st.session_state[extra_key] = 1

            num_extra = st.session_state[extra_key]
            extra_candidates = []
            tokens = current_data['sen'].split()
            max_token_idx = len(tokens) - 1

            for ei in range(num_extra):
                st.markdown(f"**补充候选 {ei + 1}**")
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    es = st.number_input(
                        "起始位置", min_value=0, max_value=max_token_idx, value=0,
                        key=f"extra_start_{st.session_state.current_idx}_{ei}"
                    )
                with col2:
                    ee = st.number_input(
                        "结束位置", min_value=0, max_value=max_token_idx, value=0,
                        key=f"extra_end_{st.session_state.current_idx}_{ei}"
                    )
                with col3:
                    if num_extra > 1:
                        if st.button("🗑️", key=f"extra_del_{st.session_state.current_idx}_{ei}"):
                            st.session_state[extra_key] -= 1
                            st.rerun()

                if es <= ee:
                    extra_text = get_span_text(current_data['sen'], es, ee + 1)
                    extra_preview = display_sentence_with_highlights(
                        current_data['sen'], prd_idx_0based, (es, ee)
                    )
                    st.markdown(f'<div class="preview-box">{extra_preview}</div>', unsafe_allow_html=True)
                    st.caption(f"📝 选中文本: '{extra_text}'")
                    extra_candidates.append({'start': es, 'end': ee, 'text': extra_text})
                else:
                    st.error(f"❌ 补充候选 {ei + 1}: 起始位置不能大于结束位置")

            if st.button("➕ 添加补充候选", key=f"add_extra_btn_{st.session_state.current_idx}"):
                st.session_state[extra_key] += 1
                st.rerun()

            # 将预设 span 和补充候选合并为 list，提交时自动走 list 分支
            if len(extra_candidates) == num_extra and all(c['start'] <= c['end'] for c in extra_candidates):
                selected_mapping = [selected_mapping] + [
                    {'start': c['start'], 'end': c['end'], 'text': c['text'], 'models': ['补充']}
                    for c in extra_candidates
                ]
    elif select_option == len(option_mappings):
        # 无论元
        st.info("🚫 该谓词无论元")
        selected_mapping = []
    else:
        # 自定义标注输入框 - 支持多候选
        st.info("💡 自定义标注：请输入论元的起始和结束位置（0-based索引）")

        # 初始化当前句子的候选数量
        custom_spans_key = f"custom_spans_{st.session_state.current_idx}"
        if custom_spans_key not in st.session_state:
            st.session_state[custom_spans_key] = 1

        num_candidates = st.session_state[custom_spans_key]
        custom_candidates = []

        # 渲染多组候选输入
        for candidate_idx in range(num_candidates):
            st.markdown(f"**候选 {candidate_idx + 1}**")
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                custom_start = st.number_input(
                    "起始位置",
                    min_value=0,
                    max_value=len(current_data['sen'].split()) - 1,
                    value=0,
                    key=f"custom_start_{st.session_state.current_idx}_{candidate_idx}"
                )
            with col2:
                custom_end = st.number_input(
                    "结束位置",
                    min_value=0,
                    max_value=len(current_data['sen'].split()) - 1,
                    value=0,
                    key=f"custom_end_{st.session_state.current_idx}_{candidate_idx}"
                )
            with col3:
                if num_candidates > 1:
                    if st.button("🗑️", key=f"delete_{st.session_state.current_idx}_{candidate_idx}"):
                        st.session_state[custom_spans_key] -= 1
                        st.rerun()

            # 预览当前候选
            if custom_start <= custom_end:
                custom_text = get_span_text(current_data['sen'], custom_start, custom_end + 1)
                custom_preview = display_sentence_with_highlights(
                    current_data['sen'],
                    prd_idx_0based,
                    (custom_start, custom_end)
                )
                st.markdown(f'<div class="preview-box">{custom_preview}</div>', unsafe_allow_html=True)
                st.caption(f"📝 选中文本: '{custom_text}'")

                custom_candidates.append({
                    'start': custom_start,
                    'end': custom_end,
                    'text': custom_text
                })
            else:
                st.error(f"❌ 候选 {candidate_idx + 1}: 起始位置不能大于结束位置")

        # 添加候选按钮
        if st.button("➕ 添加候选", key=f"add_candidate_{st.session_state.current_idx}"):
            st.session_state[custom_spans_key] += 1
            st.rerun()

        # 设置 selected_mapping 为多候选列表
        if len(custom_candidates) == num_candidates and all(c['start'] <= c['end'] for c in custom_candidates):
            selected_mapping = custom_candidates
        else:
            selected_mapping = None

    # 提交按钮用 form 包裹
    with st.form(key=f"annotation_form_{st.session_state.current_idx}"):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submitted = st.form_submit_button("✅ 提交", disabled=not grammar_check_passed, use_container_width=True, type="primary")
        with col2:
            skip_button = st.form_submit_button("⏭️ 跳过", use_container_width=True)

        if submitted and selected_mapping is not None:
            # 构建标注数据
            annotation = {
                'idx': current_data['idx'],
                'sentence': current_data['sen'],
                'prd_word': current_data['prd_word'],
                'prd_idx': current_data['prd_idx'],
                'label': current_data['label'],
                'span_mean': current_data['span_mean'],
                'type': current_data.get('type', ''),
                'timestamp': pd.Timestamp.now().isoformat(),
                'grammar_status': grammar_status,
                'grammar_error_desc': error_description if grammar_status == "有语法错误" else "",
                'selected_spans': (
                    []
                    if isinstance(selected_mapping, list) and len(selected_mapping) == 0
                    else [{'start': selected_mapping['start'],
                           'end': selected_mapping['end'],
                           'text': selected_mapping['text'],
                           'models': selected_mapping['models']}]
                    if isinstance(selected_mapping, dict)
                    else [{'start': c['start'], 'end': c['end'], 'text': c['text'], 'models': c.get('models', ['自定义'])}
                          for c in selected_mapping]
                )
            }

            # 更新或添加标注
            existing_idx = None
            for i, ann in enumerate(st.session_state.annotations):
                if ann['idx'] == current_data['idx']:
                    existing_idx = i
                    break

            if existing_idx is not None:
                st.session_state.annotations[existing_idx] = annotation
                st.toast("✅ 已更新标注", icon="✅")
            else:
                st.session_state.annotations.append(annotation)
                st.toast("✅ 标注完成", icon="✅")

            save_annotations()

            # 自动跳转到下一句
            if st.session_state.current_idx < len(st.session_state.data) - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.balloons()
                st.info("🎉 最后一句了")

        if skip_button:
            # 自动跳转到下一句
            if st.session_state.current_idx < len(st.session_state.data) - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.info("🎉 最后一句了")

    # 导航按钮
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("◀ 上一句", disabled=st.session_state.current_idx == 0, use_container_width=True):
            st.session_state.current_idx -= 1
            st.rerun()

    with col2:
        if st.button("下一句 ▶", disabled=st.session_state.current_idx >= len(st.session_state.data) - 1, use_container_width=True):
            st.session_state.current_idx += 1
            st.rerun()

    with col3:
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()

    with col4:
        if st.button("📊 结果", use_container_width=True, type="secondary"):
            st.session_state.show_results = not st.session_state.get('show_results', False)
            st.rerun()

    # 显示已标注结果
    if st.session_state.get('show_results', False) and st.session_state.annotations:
        st.markdown("**📋 已标注结果**")

        # 转换为DataFrame显示
        df_data = []
        for ann in st.session_state.annotations:
            row = {
                '序号': ann['idx'],
                '谓词': ann['prd_word'],
                '标签': ann['label'],
                '论元意思': ann['span_mean'],
                '语法状态': ann.get('grammar_status', 'N/A'),
            }

            # 兼容新格式 selected_spans 和旧格式 selected_span
            spans = ann.get('selected_spans')
            if spans is None:
                spans = [ann['selected_span']] if 'selected_span' in ann else []
            if isinstance(spans, list) and len(spans) == 0:
                row['选择的span'] = '无论元'
            elif spans:
                row['选择的span'] = ' | '.join([s['text'] for s in spans])
                row['起始位置'] = ' | '.join([str(s['start']) for s in spans])
                row['结束位置'] = ' | '.join([str(s['end']) for s in spans])
                row['来源'] = ', '.join(spans[0].get('models', []))

            df_data.append(row)

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=300)

        # 导出按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 导出JSON", use_container_width=True):
                st.json(st.session_state.annotations)
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 导出CSV",
                data=csv,
                file_name=f"annotations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
