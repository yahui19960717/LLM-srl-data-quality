"""
谓词级别 review 界面。

用途：
1. 给定当前纠正后的 final data；
2. 对随机抽样的谓词，展示 ARG0-ARG5 所有 label；
3. 对无 span 的 label，人工判断是否存在合理 span；
4. 对已有 span 的 label，人工判断是否还存在其他合理 span；
5. optional 是独立标记，不影响 span 多选；
6. 保存结果后，可计算 candidate pool 的 span-level recall。

运行方式：
1. 先生成 review 数据：
   python3 build_span_recall_review.py
2. 启动网页：
   streamlit run span_recall_review_app.py
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st


DATA_FILE = "analysis/span_recall_review_50.json"
SAVE_FILE = "analysis/span_recall_review_temp_lyh.json"
CORE_LABELS = [f"ARG{i}" for i in range(6)]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_span_text(sentence, start, end):
    """根据 token start/end 取 span 文本。这里 end 是 inclusive。"""
    tokens = sentence.split()
    if start is None or end is None:
        return ""
    if 0 <= start <= end < len(tokens):
        return " ".join(tokens[start : end + 1])
    return ""


def span_key(span):
    """用于判断人工最终 span 是否已经在候选池中。"""
    return (str(span.get("start", "")), str(span.get("end", "")), span.get("text", ""))


def display_sentence(sentence, prd_idx, selected_span=None, show_index=True):
    tokens = sentence.split()
    parts = []
    for i, tok in enumerate(tokens):
        index = f"<sub>{i}</sub>" if show_index else ""
        if i == prd_idx:
            parts.append(f'<span class="predicate">{tok}{index}</span>')
        elif selected_span and selected_span[0] <= i <= selected_span[1]:
            parts.append(f'<span class="argument">{tok}{index}</span>')
        else:
            parts.append(f"{tok}{index}")
    return " ".join(parts)


def resolve_display_prd_idx(sentence, prd_word, prd_idx):
    """兼容 0-based 和 1-based predicate index，仅用于界面高亮。"""
    tokens = sentence.split()
    if 0 <= prd_idx < len(tokens) and tokens[prd_idx] == prd_word:
        return prd_idx
    one_based = prd_idx - 1
    if 0 <= one_based < len(tokens) and tokens[one_based] == prd_word:
        return one_based
    return prd_idx


def load_review_data():
    """优先加载已有 review 进度，否则加载初始 review 数据。"""
    if os.path.exists(SAVE_FILE):
        return read_json(SAVE_FILE)
    if os.path.exists(DATA_FILE):
        return read_json(DATA_FILE)
    st.error(f"找不到数据文件：{DATA_FILE}。请先运行 python3 build_span_recall_review.py")
    return {"settings": {}, "records": []}


def save_review_data():
    st.session_state.review_data["last_index"] = st.session_state.current_idx
    st.session_state.review_data["updated_at"] = datetime.now().isoformat()
    write_json(SAVE_FILE, st.session_state.review_data)
    st.toast("已保存")


def current_saved_label(record, label_name):
    """从当前 record 中取某个 label 的状态。"""
    for label in record["labels"]:
        if label["label"] == label_name:
            return label
    return {
        "label": label_name,
        "role_description": None,
        "optional": False,
        "candidate_spans": [],
        "reviewed_spans": [],
        "no_valid_span": True,
        "review_note": "",
    }


def normalize_candidate_span(span):
    return {
        "start": span.get("start"),
        "end": span.get("end"),
        "text": span.get("text", ""),
        "models": span.get("models", []),
        "source": span.get("source", "candidate"),
    }


def make_custom_span(sentence, start, end):
    """构造人工新增 span。start/end 均为 0-based inclusive。"""
    return {
        "start": start,
        "end": end,
        "text": get_span_text(sentence, start, end),
        "models": ["human_added"],
        "source": "human_added",
    }


def init_custom_state(record_id, label_name, old_custom):
    """把已保存的人工新增 span 恢复到 session state 中。"""
    count_key = f"custom_count_{record_id}_{label_name}"
    loaded_key = f"custom_loaded_{record_id}_{label_name}"
    if st.session_state.get(loaded_key):
        return

    st.session_state[count_key] = max(1, len(old_custom))
    for i, span in enumerate(old_custom):
        st.session_state[f"custom_start_{record_id}_{label_name}_{i}"] = int(span.get("start", 0) or 0)
        st.session_state[f"custom_end_{record_id}_{label_name}_{i}"] = int(span.get("end", 0) or 0)
    st.session_state[loaded_key] = True


def calculate_span_recall(review_data):
    """
    计算 candidate pool 的 span-level recall。

    分母：人工 review 后保留的所有 valid spans。
    分子：这些 valid spans 中，原本已经出现在 candidate_spans 里的数量。
    人工新增 span 如果不在 candidate_spans 中，就说明候选池没有覆盖。
    """
    total_valid = 0
    covered = 0
    human_added = 0

    for record in review_data.get("records", []):
        for label in record.get("labels", []):
            candidate_keys = {span_key(s) for s in label.get("candidate_spans", [])}
            for span in label.get("reviewed_spans", []):
                total_valid += 1
                if span_key(span) in candidate_keys:
                    covered += 1
                if span.get("source") == "human_added" and span_key(span) not in candidate_keys:
                    human_added += 1

    recall = covered / total_valid if total_valid else 0.0
    return {
        "covered": covered,
        "total_valid": total_valid,
        "human_added_not_covered": human_added,
        "recall": recall,
    }


st.set_page_config(page_title="SRL Span Recall Review", layout="wide")
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; max-width: 100%; }
.sentence-box {
  background: #f5f7fb; border-left: 4px solid #5874c8; border-radius: 6px;
  padding: 0.8rem 1rem; line-height: 1.7; font-size: 1.05rem;
}
.predicate { color: #b42318; font-weight: 700; background: #ffe4e0; padding: 2px 4px; border-radius: 3px; }
.argument { background: #fff0b8; font-weight: 650; padding: 2px 4px; border-radius: 3px; }
.label-box { border: 1px solid #ddd; border-radius: 6px; padding: 0.8rem; margin: 0.8rem 0; background: #fff; }
.meta { color: #666; font-size: 0.9rem; }
sub { color: #777; font-size: 0.65em; margin-left: 1px; }
</style>
""",
    unsafe_allow_html=True,
)


if "review_data" not in st.session_state:
    st.session_state.review_data = load_review_data()
if "current_idx" not in st.session_state:
    st.session_state.current_idx = st.session_state.review_data.get("last_index", 0)

records = st.session_state.review_data.get("records", [])
if not records:
    st.stop()

st.sidebar.title("Review Progress")
reviewed_count = sum(1 for r in records if r.get("review_status") == "reviewed")
st.sidebar.metric("Total predicates", len(records))
st.sidebar.metric("Reviewed", reviewed_count)
st.sidebar.progress(reviewed_count / len(records))

if st.sidebar.button("Save", use_container_width=True):
    save_review_data()

if st.sidebar.button("Load saved", use_container_width=True):
    st.session_state.review_data = load_review_data()
    st.session_state.current_idx = st.session_state.review_data.get("last_index", 0)
    st.rerun()

jump_to = st.sidebar.number_input(
    "Jump to",
    min_value=1,
    max_value=len(records),
    value=st.session_state.current_idx + 1,
)
if st.sidebar.button("Go", use_container_width=True):
    st.session_state.current_idx = jump_to - 1
    st.rerun()

if st.sidebar.button("Next pending", use_container_width=True):
    for i in range(st.session_state.current_idx + 1, len(records)):
        if records[i].get("review_status") != "reviewed":
            st.session_state.current_idx = i
            st.rerun()
    for i in range(0, st.session_state.current_idx + 1):
        if records[i].get("review_status") != "reviewed":
            st.session_state.current_idx = i
            st.rerun()

if st.sidebar.button("Compute recall", use_container_width=True):
    stats = calculate_span_recall(st.session_state.review_data)
    st.sidebar.write(
        {
            "recall": f"{stats['recall'] * 100:.2f}%",
            "covered": stats["covered"],
            "total_valid": stats["total_valid"],
            "human_added_not_covered": stats["human_added_not_covered"],
        }
    )

record = records[st.session_state.current_idx]
display_prd_idx = resolve_display_prd_idx(record["sentence"], record["prd_word"], record["prd_idx"])

st.title("SRL Span Recall Review")
st.caption(
    f"{st.session_state.current_idx + 1} / {len(records)} | "
    f"domain={record['domain']} | id={record['review_id']}"
)

col1, col2, col3 = st.columns(3)
col1.metric("Predicate", record["prd_word"])
col2.metric("Predicate idx", record["prd_idx"])
col3.metric("Domain", record["domain"])

# st.markdown("完整句子")
st.markdown(
    f'<div class="sentence-box">{display_sentence(record["sentence"], display_prd_idx)}</div>',
    unsafe_allow_html=True,
)

new_labels = []

for label in record["labels"]:
    label_name = label["label"]
    candidate_spans = [normalize_candidate_span(s) for s in label.get("candidate_spans", [])]
    old_reviewed = label.get("reviewed_spans", candidate_spans)
    old_reviewed_keys = {span_key(s) for s in old_reviewed}
    old_custom = [s for s in old_reviewed if s.get("source") == "human_added"]
    init_custom_state(record["review_id"], label_name, old_custom)

    with st.container():
        st.subheader(f"{label_name}  {label.get('role_description') or ''}")

        optional = st.checkbox(
            "可标可不标 optional",
            value=bool(label.get("optional", False)),
            key=f"optional_{record['review_id']}_{label_name}",
        )
        no_valid_span = st.checkbox(
            "该 role 无合理 span",
            value=bool(label.get("no_valid_span", len(candidate_spans) == 0)),
            key=f"no_valid_{record['review_id']}_{label_name}",
        )

        selected_candidate_spans = []
        if candidate_spans:
            st.caption("候选 spans，可多选。取消勾选表示该候选不是最终有效 span。")
            for i, span in enumerate(candidate_spans):
                default = span_key(span) in old_reviewed_keys
                checked = st.checkbox(
                    f"{span.get('text', '')}  [{span.get('start')}:{span.get('end')}]  ",
                    # f"source={','.join(span.get('models', []))}",
                    value=default,
                    key=f"cand_{record['review_id']}_{label_name}_{i}",
                )
                if checked:
                    selected_candidate_spans.append(span)
                    try:
                        start = int(span["start"])
                        end = int(span["end"])
                        st.markdown(
                            f'<div class="sentence-box">{display_sentence(record["sentence"], display_prd_idx, (start, end), show_index=False)}</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        pass
        else:
            st.caption("当前没有候选 span。如果该 role 存在合理 span，请在下面自定义添加。")

        add_custom = st.checkbox(
            "自定义标注",
            value=bool(old_custom),
            key=f"add_custom_{record['review_id']}_{label_name}",
        )
        custom_spans = []
        if add_custom:
            st.info("请输入论元的起始和结束位置（0-based 索引，结束位置为 inclusive）")
            count_key = f"custom_count_{record['review_id']}_{label_name}"
            num_candidates = st.session_state.get(count_key, 1)
            max_token_idx = len(record["sentence"].split()) - 1

            for candidate_idx in range(num_candidates):
                st.markdown(f"**候选 {candidate_idx + 1}**")
                col_start, col_end, col_delete = st.columns([2, 2, 1])
                start_key = f"custom_start_{record['review_id']}_{label_name}_{candidate_idx}"
                end_key = f"custom_end_{record['review_id']}_{label_name}_{candidate_idx}"

                with col_start:
                    custom_start = st.number_input(
                        "起始位置",
                        min_value=0,
                        max_value=max_token_idx,
                        value=int(st.session_state.get(start_key, 0)),
                        key=start_key,
                    )
                with col_end:
                    custom_end = st.number_input(
                        "结束位置",
                        min_value=0,
                        max_value=max_token_idx,
                        value=int(st.session_state.get(end_key, 0)),
                        key=end_key,
                    )
                with col_delete:
                    if num_candidates > 1 and st.button(
                        "删除",
                        key=f"delete_custom_{record['review_id']}_{label_name}_{candidate_idx}",
                    ):
                        st.session_state[count_key] = max(1, num_candidates - 1)
                        st.rerun()

                if custom_start <= custom_end:
                    custom_span = make_custom_span(record["sentence"], custom_start, custom_end)
                    custom_spans.append(custom_span)
                    st.markdown(
                        f'<div class="sentence-box">{display_sentence(record["sentence"], display_prd_idx, (custom_start, custom_end), show_index=False)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"选中文本: {custom_span['text']}")
                else:
                    st.error("起始位置不能大于结束位置")

            if st.button("添加候选", key=f"add_custom_candidate_{record['review_id']}_{label_name}"):
                st.session_state[count_key] = num_candidates + 1
                st.rerun()

        reviewed_spans = [] if no_valid_span else selected_candidate_spans + custom_spans
        new_labels.append(
            {
                "label": label_name,
                "role_description": label.get("role_description"),
                "optional": optional,
                "candidate_spans": candidate_spans,
                "reviewed_spans": reviewed_spans,
                "no_valid_span": no_valid_span,
                "review_note": label.get("review_note", ""),
            }
        )


left, mid, right, extra = st.columns(4)
with left:
    if st.button("Previous", disabled=st.session_state.current_idx == 0, use_container_width=True):
        st.session_state.current_idx -= 1
        st.rerun()
with mid:
    if st.button("Save current", type="primary", use_container_width=True):
        record["labels"] = new_labels
        record["review_status"] = "reviewed"
        record["reviewed_at"] = datetime.now().isoformat()
        save_review_data()
with right:
    if st.button("Save & next", type="primary", use_container_width=True):
        record["labels"] = new_labels
        record["review_status"] = "reviewed"
        record["reviewed_at"] = datetime.now().isoformat()
        save_review_data()
        if st.session_state.current_idx < len(records) - 1:
            st.session_state.current_idx += 1
        st.rerun()
with extra:
    if st.button("Next", disabled=st.session_state.current_idx >= len(records) - 1, use_container_width=True):
        st.session_state.current_idx += 1
        st.rerun()

if st.checkbox("Show reviewed table"):
    rows = []
    for r in records:
        for label in r.get("labels", []):
            for span in label.get("reviewed_spans", []):
                rows.append(
                    {
                        "review_id": r["review_id"],
                        "domain": r["domain"],
                        "prd_word": r["prd_word"],
                        "label": label["label"],
                        "optional": label.get("optional", False),
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "text": span.get("text"),
                        "source": span.get("source"),
                    }
                )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
