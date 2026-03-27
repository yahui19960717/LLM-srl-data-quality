"""
在llamafactory的环境中运行（有streamlit包)
审核界面，对于一个例子，可以看到两个标注者的标注结果、句子、需要标注的index等
"""
import os
import json
import copy
import streamlit as st
from typing import Dict, List, Optional, Tuple
# annotation/analysis/bn/annotators_analysis_smallmodel163.json 
# annotation/analysis/bn/annotators_analysis_smallmodel163.json
# ──────────────────────────────────────────────
# 硬编码文件路径（按需修改）
# ──────────────────────────────────────────────
# DATA_FILE   = "analysis/bn/annotators_analysis_smallmodel163.json"
# DATA_FILE   = "analysis/bn/annotators_analysis_gold113.json" 
# DATA_FILE = "analysis/bn/annotators_analysis_single_gold_o1wrongdsright_93.json"      # extract_disagreements.py 生成的文件
# DATA_FILE   = "analysis/bn/annotators_analysis_goldrandom30.json"   
# DATA_FILE   = "analysis/bn/annotators_analysis_o1right_random30.json" 
# DATA_FILE   = "analysis/bn/annotators_analysis_botherror_random30.json" 
# DATA_FILE = "analysis/bn/annotators_analysis_ssmallmodel_overlap_179.json"
# OUTPUT_FILE = "analysis/bn/temp.json"          # 裁判结果保存路径

# DATA_FILE = "analysis/bn/annotators_analysis_smnotrecall_21.json"
# DATA_FILE = "analysis/bn/annotators_analysis_smnotrecallright_random30.json"
# DATA_FILE = "analysis/bn/annotators_analysis_smnotrecallright_110.json"
# OUTPUT_FILE = "analysis/bn/temp.json" 
### tc
# DATA_FILE = "analysis/tc/annotators_tc_single_gold_201.json"
# DATA_FILE = "analysis/tc/annotators_tc_smallmodel_removerepeate_322.json"
DATA_FILE = "analysis/tc/annotators_tc_o1wrong_dsright_80.json"
OUTPUT_FILE = "analysis/tc/temp.json" 
# ──────────────────────────────────────────────
# 颜色常量
# ──────────────────────────────────────────────
COL_A        = "#FCA5A5"
COL_A_TXT    = "#7F1D1D"
COL_B        = "#93C5FD"
COL_B_TXT    = "#1E3A5F"
COL_BOTH     = "#6EE7B7"
COL_BOTH_TXT = "#064E3B"
COL_ADJ      = "#FDE68A"
COL_ADJ_TXT  = "#78350F"

MATCH_LABEL = {
    "partial": "部分匹配",
    "none":    "完全不匹配",
    "empty_a": "仅B有标注",
    "empty_b": "仅A有标注",
}

# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="标注一致性审核",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }

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

    .sentence-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 1.05rem;
        line-height: 1.8;
        margin: 0.4rem 0;
        border-left: 3px solid #667eea;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    .preview-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 1rem;
        line-height: 1.6;
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


    .info-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-radius: 6px;
        padding: 0.4rem 0.7rem;
        margin: 0.2rem 0;
        border-left: 2px solid #00acc1;
        font-size: 0.9rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-done    { background: linear-gradient(120deg,#d4edda,#c3e6cb); color:#155724; border:1px solid #c3e6cb; }
    .status-pending { background: linear-gradient(120deg,#fff3cd,#ffeaa7); color:#856404; border:1px solid #ffeaa7; }

    .match-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.73rem;
        font-weight: 700;
        margin-right: 5px;
    }
    .badge-partial  { background:#EDE9FE; color:#6D28D9; }
    .badge-none     { background:#FEE2E2; color:#B91C1C; }
    .badge-empty_a  { background:#DBEAFE; color:#1D4ED8; }
    .badge-empty_b  { background:#FEF3C7; color:#B45309; }
    .badge-done     { background:#D1FAE5; color:#065F46; }
    .badge-meta     { background:#F1F5F9; color:#475569; }

    .span-pill {
        display: inline-block;
        border-radius: 20px;
        padding: 2px 11px;
        font-size: 0.83rem;
        font-weight: 600;
        margin: 2px 3px;
    }

    .sec-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: .07em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
        padding: 0.3rem 0.8rem !important;
        font-size: 0.9rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    hr { margin: 0.6rem 0 !important; border:none; height:1px;
         background: linear-gradient(90deg, transparent, #667eea, transparent); }

    .stAlert { padding: 0.4rem 0.8rem !important; font-size: 0.85rem !important; }
    .stMarkdown { margin-bottom: 0.3rem !important; }
    h1,h2,h3,h4 { margin-top:0.4rem !important; margin-bottom:0.3rem !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 数据加载 & 保存
# ──────────────────────────────────────────────

def load_data():
    """加载 disagreements.json，写入 session_state"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            st.session_state.raw      = raw
            st.session_state.meta     = raw.get("meta", {})
            st.session_state.records  = copy.deepcopy(raw.get("records", []))
            st.success(f"✅ 成功加载 {len(st.session_state.records)} 条不一致数据")
        except Exception as e:
            st.error(f"❌ 加载失败: {e}")
            st.session_state.records = []
    else:
        st.error(f"❌ 文件不存在: {DATA_FILE}")
        st.session_state.records = []


def save_data():
    """将当前裁判结果保存到 OUTPUT_FILE"""
    try:
        out = copy.deepcopy(st.session_state.raw)
        out["records"] = st.session_state.records
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        st.success(f"💾 已保存至 {OUTPUT_FILE}")
    except Exception as e:
        st.error(f"❌ 保存失败: {e}")


# ──────────────────────────────────────────────
# session_state 初始化
# ──────────────────────────────────────────────

if 'records' not in st.session_state:
    if os.path.exists(DATA_FILE):
        load_data()
    else:
        st.session_state.records = []
        st.session_state.meta    = {}
        st.session_state.raw     = {}

if 'show_filter' not in st.session_state:
    st.session_state.show_filter = True

if 'current_idx' not in st.session_state:   # ← 加这两行
    st.session_state.current_idx = 0
# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

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


def highlight_sentence(sentence: str,
                        spans_a: List[Dict],
                        spans_b: List[Dict],
                        adj_spans: Optional[List[Dict]] = None) -> str:
    n = len(sentence)
    tag_a   = [False] * n
    tag_b   = [False] * n
    tag_adj = [False] * n

    for s in spans_a:
        for i in range(s['start'], min(s['end'] + 1, n)):
            tag_a[i] = True
    for s in spans_b:
        for i in range(s['start'], min(s['end'] + 1, n)):
            tag_b[i] = True
    if adj_spans:
        for s in adj_spans:
            for i in range(s['start'], min(s['end'] + 1, n)):
                tag_adj[i] = True

    html = ""
    i = 0
    while i < n:
        a, b, d = tag_a[i], tag_b[i], tag_adj[i]
        if not a and not b and not d:
            html += sentence[i]
            i += 1
            continue

        if a and b:
            bg, fg, tip = COL_BOTH, COL_BOTH_TXT, "双方一致"
        elif a:
            bg, fg, tip = COL_A, COL_A_TXT, "仅A"
        else:
            bg, fg, tip = COL_B, COL_B_TXT, "仅B"

        j = i + 1
        while j < n and tag_a[j] == a and tag_b[j] == b and tag_adj[j] == d:
            j += 1
        chunk = sentence[i:j]
        border = f"outline:2px solid {COL_ADJ};" if d else ""
        html += (f'<span title="{tip}" style="background:{bg};color:{fg};'
                 f'border-radius:4px;padding:1px 3px;font-weight:700;{border}">'
                 f'{chunk}</span>')
        i = j
    return html


def parse_manual_spans(sentence: str, raw: str) -> Tuple[List[Dict], str]:
    spans, errors = [], []
    for part in raw.replace('，', ',').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            s, e = part.split('-')
            s, e = int(s.strip()), int(e.strip())
            if s < 0 or e >= len(sentence) or s > e:
                errors.append(f"{s}-{e} 超出范围")
                continue
            spans.append({"start": s, "end": e, "text": sentence[s:e+1]})
        except Exception:
            errors.append(f'"{part}" 格式有误')
    return spans, "；".join(errors)


def spans_to_display(spans: List[Dict], bg: str, fg: str) -> str:
    if not spans:
        return '<span style="color:#94A3B8;font-size:0.85rem;">（未标注）</span>'
    return "".join(
        f'<span class="span-pill" style="background:{bg};color:{fg};">'
        f'{s["text"]} <span style="opacity:.6;font-size:.75rem;">[{s["start"]}-{s["end"]}]</span>'
        f'</span>'
        for s in spans
    )


# ──────────────────────────────────────────────
# 侧边栏
# ──────────────────────────────────────────────

def sidebar(records: List[Dict], meta: Dict):
    with st.sidebar:
        st.markdown("**⚖️ 标注一致性审核**")

        # 加载 / 保存
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 重新加载", use_container_width=True, type="secondary"):
                load_data()
                st.rerun()
        with col2:
            if st.button("💾 保存", use_container_width=True, type="primary"):
                save_data()

        # 导出
        if records:
            out_bytes = json.dumps(
                {**st.session_state.raw, "records": records},
                ensure_ascii=False, indent=2
            ).encode("utf-8")
            st.download_button(
                "⬇️ 导出 JSON",
                data=out_bytes,
                file_name="adjudicated.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")

        # 进度统计
        total = len(records)
        done  = sum(1 for r in records if r.get("adjudicated_spans") is not None)
        st.markdown("**📊 进度**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总计", total)
        with col2:
            st.metric("已裁判", done)
        st.progress(done / total if total else 0, text=f"{done/total*100:.1f}%" if total else "0%")

        st.markdown("---")

        # 筛选
        st.markdown("**🎛️ 筛选**")
        name_a = meta.get("name_a", "标注员A")
        name_b = meta.get("name_b", "标注员B")
        all_labels = sorted(set(r['label'] for r in records)) if records else []

        sel_types = st.multiselect(
            "匹配类型",
            options=["partial", "none", "empty_a", "empty_b"],
            default=["partial", "none", "empty_a", "empty_b"],
            format_func=lambda x: MATCH_LABEL[x],
        )
        sel_labels  = st.multiselect("语义角色标签", options=all_labels, default=all_labels)
        adj_filter  = st.selectbox("裁判状态", ["全部", "待裁判", "已裁判"])
        search      = st.text_input("🔍 搜索句子 / 谓词", placeholder="输入关键字...")

        st.markdown("---")

        # 图例
        st.markdown("**🎨 图例**")
        st.markdown(f"""
        <span class="span-pill" style="background:{COL_A};color:{COL_A_TXT};">仅 {name_a}</span>
        <span class="span-pill" style="background:{COL_B};color:{COL_B_TXT};">仅 {name_b}</span><br>
        <span class="span-pill" style="background:{COL_BOTH};color:{COL_BOTH_TXT};">双方一致</span>
        <span class="span-pill" style="background:{COL_ADJ};color:{COL_ADJ_TXT};">裁判结果</span>
        """, unsafe_allow_html=True)

    return name_a, name_b, sel_types, sel_labels, adj_filter, search


# ──────────────────────────────────────────────
# 单条记录渲染
# ──────────────────────────────────────────────

def render_record(rec: Dict, name_a: str, name_b: str):
    idx       = rec['idx']
    sentence  = rec['sentence']
    spans_a   = rec.get('spans_a', [])
    spans_b   = rec.get('spans_b', [])
    m_type    = rec['match_type']
    adj_spans = rec.get('adjudicated_spans')
    adj_note  = rec.get('adjudication_note', '')
    is_done   = adj_spans is not None

    badge_cls = "badge-done" if is_done else f"badge-{m_type}"
    badge_txt = "已裁判"     if is_done else MATCH_LABEL.get(m_type, m_type)
    status_cls = "status-done" if is_done else "status-pending"
    status_txt = "✅ 已裁判"  if is_done else "⏳ 待裁判"
    score_html = (f'<span class="match-badge badge-meta">覆盖率 {rec["score"]:.0%}</span>'
                  if m_type == "partial" else "")

    highlighted = highlight_sentence(sentence, spans_a, spans_b, adj_spans if is_done else None)

    # ── 标题行
    col_h1, col_h2 = st.columns([5, 1])
    with col_h1:
        st.markdown(
            f'<span class="match-badge {badge_cls}">{badge_txt}</span>'
            f'<span class="match-badge badge-meta">#{idx}</span>'
            f'<span class="match-badge badge-meta">{rec["label"]}</span>'
            f'{score_html}',
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown(f'<span class="status-badge {status_cls}">{status_txt}</span>',
                    unsafe_allow_html=True)

    # ── 元信息行
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="info-card"><b>谓词:</b> {rec["prd_word"]}</div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="info-card"><b>标签:</b> {rec["label"]}</div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="info-card"><b>论元意思:</b> {rec.get("span_mean","")}</div>',
                    unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="info-card"><b>类型:</b> {m_type}</div>',
                    unsafe_allow_html=True)

    # ── 句子高亮
    html_sentence = display_sentence_with_highlights(
        sentence,
        
        rec['prd_idx']-1,
        selected_span=None,
        show_index=True
    )
    # st.markdown(f'<div class="sentence-box">{highlighted}</div>', unsafe_allow_html=True)
    st.markdown(f'📒 {html_sentence}', unsafe_allow_html=True)
    st.markdown(f'📋  {sentence}', unsafe_allow_html=True)

    # ── 双方标注对比
    col_a, col_b, col_adj = st.columns(3)
    with col_a:
        st.markdown(f'<div class="sec-title">🔴 {name_a}</div>', unsafe_allow_html=True)
        st.markdown(spans_to_display(spans_a, COL_A, COL_A_TXT), unsafe_allow_html=True)
        if rec.get('optional_a'):
            st.caption("⚠️ 可标可不标")
    with col_b:
        st.markdown(f'<div class="sec-title">🔵 {name_b}</div>', unsafe_allow_html=True)
        st.markdown(spans_to_display(spans_b, COL_B, COL_B_TXT), unsafe_allow_html=True)
        if rec.get('optional_b'):
            st.caption("⚠️ 可标可不标")
    with col_adj:
        st.markdown('<div class="sec-title">⚖️ 裁判结果</div>', unsafe_allow_html=True)
        st.markdown(spans_to_display(adj_spans or [], COL_ADJ, COL_ADJ_TXT),
                    unsafe_allow_html=True)

    # ── 裁判编辑区
    exp_label = f"{'✏️ 修改裁判结果' if is_done else '⚖️ 填写裁判结果'}（#{idx}）"
    with st.expander(exp_label, expanded=False):

        # 候选 span 下拉
        all_candidate_spans: List[Dict] = []
        seen_keys = set()
        for s in spans_a + spans_b:
            k = (s['start'], s['end'])
            if k not in seen_keys:
                seen_keys.add(k)
                all_candidate_spans.append(s)
        all_candidate_spans.sort(key=lambda x: x['start'])
        candidate_options = [f"{s['text']} [{s['start']}-{s['end']}]" for s in all_candidate_spans]

        st.markdown("**方式一：从候选 span 中选择**")
        if candidate_options:
            default_sel = []
            if adj_spans:
                adj_keys = {(s['start'], s['end']) for s in adj_spans}
                default_sel = [opt for opt, sp in zip(candidate_options, all_candidate_spans)
                               if (sp['start'], sp['end']) in adj_keys]
            chosen_opts = st.multiselect(
                "选择 span（可多选）",
                options=candidate_options,
                default=default_sel,
                key=f"sel_{idx}",
            )
            chosen_spans = [sp for opt, sp in zip(candidate_options, all_candidate_spans)
                            if opt in chosen_opts]
        else:
            chosen_spans = []
            st.info("双方均无 span，请使用方式二手动输入。")

        # 手动输入
        st.markdown("**方式二：手动输入 start-end（逗号分隔，如 `0-3, 7-10`）**")
        manual_input = st.text_input(
            "手动输入（留空则以方式一为准）",
            value="",
            placeholder="例：0-3, 7-10",
            key=f"manual_{idx}",
        )
        manual_spans, manual_err = [], ""
        if manual_input.strip():
            manual_spans, manual_err = parse_manual_spans(sentence, manual_input)
            if manual_err:
                st.error(f"❌ 输入有误：{manual_err}")

        final_spans = manual_spans if manual_input.strip() and not manual_err else chosen_spans

        # 实时预览
        if final_spans:
            preview_html = highlight_sentence(sentence, spans_a, spans_b, final_spans)
            st.markdown("**预览（黄色边框 = 裁判结果）**")
            st.markdown(f'<div class="preview-box">{preview_html}</div>', unsafe_allow_html=True)

        note_val = st.text_input("备注（可选）", value=adj_note, key=f"note_{idx}",
                                  placeholder="填写裁判依据或说明...")

        col_save, col_clear, _ = st.columns([1, 1, 2])
        with col_save:
            if st.button("✅ 保存", key=f"save_{idx}", use_container_width=True, type="primary"):
                for r in st.session_state.records:
                    if r['idx'] == idx:
                        r['adjudicated_spans'] = final_spans
                        r['adjudication_note'] = note_val
                        break
                st.toast("✅ 已保存", icon="✅")
                st.rerun()
        with col_clear:
            if st.button("🗑️ 清除", key=f"clear_{idx}", use_container_width=True):
                for r in st.session_state.records:
                    if r['idx'] == idx:
                        r['adjudicated_spans'] = None
                        r['adjudication_note'] = ""
                        break
                st.rerun()

    st.markdown("---")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    st.markdown('<h1 class="main-title">⚖️ 标注一致性审核工具</h1>', unsafe_allow_html=True)

    records = st.session_state.records
    meta    = st.session_state.get("meta", {})

    if not records:
        st.warning(f"⚠️ 暂无数据，请确认 `{DATA_FILE}` 文件存在并点击侧边栏「重新加载」。")

    name_a, name_b, sel_types, sel_labels, adj_filter, search = sidebar(records, meta)

    if not records:
        st.stop()

    # ── 过滤
    filtered = [
        r for r in records
        if r['match_type'] in sel_types
        and r['label'] in sel_labels
        and (not search or search in r['sentence'] or search in r['prd_word'])
        and (adj_filter == "全部"
             or (adj_filter == "已裁判" and r.get("adjudicated_spans") is not None)
             or (adj_filter == "待裁判" and r.get("adjudicated_spans") is None))
    ]

    st.markdown(f"**显示 {len(filtered)} / {len(records)} 条**")

    if not filtered:
        st.warning("没有符合条件的记录。")
        st.stop()

    # for rec in filtered:
    #     render_record(rec, name_a, name_b)
    # 边界保护
    if st.session_state.current_idx >= len(filtered):
        st.session_state.current_idx = 0

    st.markdown(f"**{st.session_state.current_idx + 1} / {len(filtered)} 条**")

    # 渲染当前条
    render_record(filtered[st.session_state.current_idx], name_a, name_b)

    # 导航按钮
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("◀ 上一条", disabled=st.session_state.current_idx == 0, use_container_width=True):
            st.session_state.current_idx -= 1
            st.rerun()
    with col2:
        if st.button("下一条 ▶", disabled=st.session_state.current_idx >= len(filtered) - 1, use_container_width=True):
            st.session_state.current_idx += 1
            st.rerun()
    with col3:
        jump = st.number_input("跳转到", min_value=1, max_value=len(filtered),
                                value=st.session_state.current_idx + 1,
                                label_visibility="collapsed")
        if jump - 1 != st.session_state.current_idx:
            st.session_state.current_idx = jump - 1
            st.rerun()


if __name__ == "__main__":
    main()
