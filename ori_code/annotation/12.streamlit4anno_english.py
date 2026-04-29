"""
Annotation Interface
Can select spans (multi-select), or customize
Optional annotation as an independent flag, does not affect span selection
"""
import os
import streamlit as st
import pandas as pd
import json
import pickle
from collections import defaultdict
from datetime import datetime
import re

# Data file path tc
domain = "tc"
DATA_FILE = f"anno/tc/annotation_tc_o1wrong_dswrong_random30.pkl"
ANNOTATION_SAVE_FILE = f"anno/{domain}/annotation_{domain}_o1wrong_dswrong_random30_lyh.json"

def read_pickle(file):
    """Read pickle data from file"""
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(len(data))
    return data


# Page configuration
st.set_page_config(
    page_title="Semantic Role Annotation Tool",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - compact layout
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
        line-height: 1.6;
        margin: 0.4rem 0;
        border-left: 3px solid #667eea;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
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
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .info-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
        margin: 0.3rem 0;
        border-left: 2px solid #00acc1;
        font-size: 0.9rem;
    }
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
    hr {
        margin: 0.8rem 0 !important;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    sub {
        color: #6c757d;
        font-size: 0.65em;
        font-weight: 500;
        margin-left: 2px;
    }
    .stMarkdown { margin-bottom: 0.3rem !important; }
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
    }
    .stRadio > label, .stTextArea > label, .stNumberInput > label {
        font-size: 0.9rem !important;
        margin-bottom: 0.2rem !important;
    }
    .stForm { border: none !important; padding: 0.5rem 0 !important; }
    .stAlert { padding: 0.4rem 0.8rem !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🏷️ Semantic Role Annotation Tool</h1>', unsafe_allow_html=True)

# Initialize session state
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'data' not in st.session_state:
    if os.path.exists(DATA_FILE):
        try:
            st.session_state.data = read_pickle(DATA_FILE)
            st.success(f"✅ Successfully loaded {len(st.session_state.data)} records")
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
            st.session_state.data = []
    else:
        st.error(f"❌ Data file not found: {DATA_FILE}")
        st.session_state.data = []


def save_annotations():
    with open(ANNOTATION_SAVE_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'annotations': st.session_state.annotations,
            'last_index': st.session_state.current_idx
        }, f, ensure_ascii=False, indent=2)
    st.success("💾 Annotations saved")


def load_annotations():
    if os.path.exists(ANNOTATION_SAVE_FILE):
        try:
            with open(ANNOTATION_SAVE_FILE, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                st.session_state.annotations = saved_data.get('annotations', [])
                st.session_state.current_idx = saved_data.get('last_index', 0)
            st.success(f"📂 Progress loaded, currently at record {st.session_state.current_idx + 1}")
            return True
        except Exception as e:
            st.error(f"❌ Failed to load annotation file: {e}")
    return False


def get_span_text(sentence, start, end):
    tokens = sentence.split()
    if start <= end and 0 <= start < len(tokens) and 0 < end <= len(tokens):
        return ' '.join(tokens[start:end])
    return ""


def display_sentence_with_highlights(sentence, prd_idx, selected_span=None, show_index=False):
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
    for ann in st.session_state.annotations:
        if ann['idx'] == current_idx:
            return ann
    return None


def slide_bar(st):
    with st.sidebar:
        st.markdown("**📊 Progress**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Load", use_container_width=True, type="secondary"):
                load_annotations()
                st.rerun()
        with col2:
            if st.button("💾 Save", use_container_width=True, type="primary"):
                save_annotations()

        total_sentences = len(st.session_state.data)
        annotated_indices = {ann['idx'] for ann in st.session_state.annotations}
        annotated_count = len(annotated_indices)
        optional_count = sum(1 for ann in st.session_state.annotations if ann.get('optional'))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total_sentences)
        with col2:
            st.metric("Annotated", annotated_count)

        progress = annotated_count / total_sentences if total_sentences > 0 else 0
        st.progress(progress, text=f"{progress*100:.1f}%")

        if optional_count > 0:
            st.caption(f"⚠️ Optional annotations: {optional_count}")

        st.markdown("---")
        st.markdown("**🔍 Jump To**")
        jump_idx = st.number_input(
            f"Index (1-{total_sentences})",
            min_value=1,
            max_value=total_sentences,
            value=st.session_state.current_idx + 1,
            key="jump_input",
            label_visibility="collapsed"
        )
        if st.button("🎯 Jump", use_container_width=True, type="primary"):
            st.session_state.current_idx = jump_idx - 1
            st.rerun()

        st.markdown("---")
        st.markdown("**⚡ Shortcuts**")
        if st.button("⏭️ Next Unannotated", use_container_width=True):
            for i in range(st.session_state.current_idx + 1, len(st.session_state.data)):
                if st.session_state.data[i]['idx'] not in annotated_indices:
                    st.session_state.current_idx = i
                    st.rerun()
                    break
            else:
                st.warning("⚠️ No unannotated records remaining")

        if st.button("⏮️ First Unannotated", use_container_width=True):
            for i in range(len(st.session_state.data)):
                if st.session_state.data[i]['idx'] not in annotated_indices:
                    st.session_state.current_idx = i
                    st.rerun()
                    break
            else:
                st.info("🎉 All records annotated")


def main_top(st, current_data, annotation_status):
    col_header1, col_header2 = st.columns([4, 1])
    with col_header1:
        st.markdown(f"**📝 Sentence {st.session_state.current_idx + 1} / {len(st.session_state.data)}**")
    with col_header2:
        if annotation_status:
            st.markdown('<span class="status-badge status-annotated">✅ Annotated</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-pending">⏳ Pending</span>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="info-card"><b>Index:</b> {current_data["idx"]}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="info-card"><b>Predicate:</b> {current_data["prd_word"]}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="info-card"><b>Label:</b> {current_data["label"]}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="info-card"><b>Argument Meaning:</b> {current_data["span_mean"]}</div>', unsafe_allow_html=True)


def grammar(current_data, st):
    st.markdown("**🔍 Sentence Check**")
    current_idx = current_data['idx']

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
            "Does the sentence have grammatical errors?",
            options=["✅ No grammatical errors", "❌ Has grammatical errors"],
            index=0 if saved_grammar is None or saved_grammar == "No grammatical errors" else 1,
            key=f"grammar_radio_{current_idx}",
            label_visibility="collapsed"
        )
    with col2:
        if grammar_status == "❌ Has grammatical errors":
            error_description = st.text_area(
                "Please describe the grammatical error:",
                value=saved_error_desc,
                key=f"grammar_desc_{current_idx}",
                placeholder="e.g., missing subject, collocation error, word order issue, etc.",
                height=60,
                label_visibility="collapsed"
            )
            if error_description and error_description.strip():
                st.success("✓ Marked as sentence error", icon="✅")
                grammar_check_passed = True
            else:
                st.error("❌ Please describe the error")
                grammar_check_passed = False
        else:
            error_description = ""
            st.success("✓ Sentence has no errors", icon="✅")
            grammar_check_passed = True

    grammar_status_clean = grammar_status.replace("✅ ", "").replace("❌ ", "")
    return grammar_status_clean, error_description, grammar_check_passed


def main():
    if not st.session_state.data:
        st.error("❌ No data loaded, please check the file path")
        return

    slide_bar(st)

    current_data = st.session_state.data[st.session_state.current_idx]
    annotation_status = get_annotation_status(current_data['idx'])

    main_top(st, current_data, annotation_status)

    # Display full sentence
    st.markdown("**📄 Full Sentence**")
    temp_sen = " ".join(current_data['sen'].split())
    # st.markdown(current_data['idx'], unsafe_allow_html=True)
    st.markdown(temp_sen, unsafe_allow_html=True)
    prd_idx_0based = current_data['prd_idx'] - 1
    html_sentence = display_sentence_with_highlights(
        current_data['sen'],
        prd_idx_0based,
        selected_span=None,
        show_index=True
    )
    st.markdown(f'<div class="sentence-box">{html_sentence}</div>', unsafe_allow_html=True)

    # Grammar check
    grammar_status, error_description, grammar_check_passed = grammar(current_data, st)

    # ── Argument Annotation Area ──────────────────────────────────────────
    st.markdown("**🎯 Annotate Argument**")

    # Build preset options
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
                    'end': end - 1,
                    'text': span_text,
                    'models': models,
                    'highlighted_sentence': highlighted_sentence
                })

    # Restore saved selection state
    saved_preset_defaults = []
    saved_no_arg = False
    saved_add_custom = False
    saved_optional = False
    if annotation_status:
        saved_spans = annotation_status.get('selected_spans', [])
        saved_optional = annotation_status.get('optional', False)
        if isinstance(saved_spans, list) and len(saved_spans) == 0:
            saved_no_arg = not annotation_status.get('has_custom', False)
        else:
            for sp in (saved_spans or []):
                if sp.get('models') == ['Custom']:
                    saved_add_custom = True
                else:
                    for i, m in enumerate(option_mappings):
                        if m['start'] == sp['start'] and m['end'] == sp['end']:
                            saved_preset_defaults.append(i)
                            break

    # ① Preset span multi-select
    preset_labels = [f"'{d['text']}'" for i, d in enumerate(option_mappings)]

    if preset_labels:
        st.caption("📋 Preset candidate spans (multi-select allowed):")
        selected_presets = []
        for i, d in enumerate(option_mappings):
            default_checked = i in saved_preset_defaults
            checked = st.checkbox(
                f"'{d['text']}'",
                value=default_checked,
                key=f"preset_{st.session_state.current_idx}_{i}"
            )
            if checked:
                selected_presets.append(i)
                # Real-time preview of selected span
                st.markdown(f'<div class="preview-box">{d["highlighted_sentence"]}</div>', unsafe_allow_html=True)
                st.caption(f"Source: {', '.join(d['models'])}  |  Position: [{d['start']}:{d['end']}]")
    else:
        selected_presets = []
        st.caption("(No preset candidate spans)")

    st.markdown("---")

    # ② Three independent options stacked vertically: Custom / Optional / No Argument
    add_custom = st.checkbox(
        "✏️ Custom Annotation",
        value=saved_add_custom,
        key=f"add_custom_{st.session_state.current_idx}",
        help="Manually enter the start and end positions of the argument"
    )
    is_optional = st.checkbox(
        "⚠️ Optional Annotation",
        value=saved_optional,
        key=f"optional_check_{st.session_state.current_idx}",
        help="Check to mark this record as 'optional'; it will be handled separately during evaluation"
    )
    no_arg = st.checkbox(
        "🚫 No Argument",
        value=saved_no_arg,
        key=f"no_arg_{st.session_state.current_idx}",
        help="This predicate has no corresponding argument in this sentence"
    )

    # Status hint
    if no_arg:
        st.info("🚫 Marked as no argument — other span selections will be ignored on submit")

    # ③ Custom annotation input (controlled by checkbox)
    custom_candidates = []
    if add_custom:
        st.markdown("**✏️ Custom Annotation**")
        st.info("💡 Enter the start and end positions of the argument (0-based index)")

        custom_spans_key = f"custom_spans_{st.session_state.current_idx}"
        if custom_spans_key not in st.session_state:
            st.session_state[custom_spans_key] = 1

        num_candidates = st.session_state[custom_spans_key]
        tokens = current_data['sen'].split()
        max_token_idx = len(tokens) - 1

        for candidate_idx in range(num_candidates):
            st.markdown(f"**Candidate {candidate_idx + 1}**")
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                custom_start = st.number_input(
                    "Start position", min_value=0, max_value=max_token_idx, value=0,
                    key=f"custom_start_{st.session_state.current_idx}_{candidate_idx}"
                )
            with col2:
                custom_end = st.number_input(
                    "End position", min_value=0, max_value=max_token_idx, value=0,
                    key=f"custom_end_{st.session_state.current_idx}_{candidate_idx}"
                )
            with col3:
                if num_candidates > 1:
                    if st.button("🗑️", key=f"delete_{st.session_state.current_idx}_{candidate_idx}"):
                        st.session_state[custom_spans_key] -= 1
                        st.rerun()

            if custom_start <= custom_end:
                custom_text = get_span_text(current_data['sen'], custom_start, custom_end + 1)
                custom_preview = display_sentence_with_highlights(
                    current_data['sen'], prd_idx_0based, (custom_start, custom_end)
                )
                st.markdown(f'<div class="preview-box">{custom_preview}</div>', unsafe_allow_html=True)
                st.caption(f"📝 Selected text: '{custom_text}'")
                custom_candidates.append({
                    'start': custom_start,
                    'end': custom_end,
                    'text': custom_text
                })
            else:
                st.error(f"❌ Candidate {candidate_idx + 1}: Start position cannot exceed end position")

        if st.button("➕ Add Candidate", key=f"add_candidate_{st.session_state.current_idx}"):
            st.session_state[custom_spans_key] += 1
            st.rerun()

    # ── Submit Area ─────────────────────────────────────────────
    with st.form(key=f"annotation_form_{st.session_state.current_idx}"):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            submitted = st.form_submit_button(
                "✅ Submit",
                disabled=not grammar_check_passed,
                use_container_width=True,
                type="primary"
            )
        with col2:
            skip_button = st.form_submit_button("⏭️ Skip", use_container_width=True)

        if submitted:
            # Aggregate final spans
            if no_arg:
                # No argument takes priority — ignore all other selections
                final_spans = []
                has_custom = False
            else:
                final_spans = []
                # Preset spans
                for idx in selected_presets:
                    m = option_mappings[idx]
                    final_spans.append({
                        'start': m['start'],
                        'end': m['end'],
                        'text': m['text'],
                        'models': m['models']
                    })
                # Custom spans
                for c in custom_candidates:
                    final_spans.append({
                        'start': c['start'],
                        'end': c['end'],
                        'text': c['text'],
                        'models': ['Custom']
                    })
                has_custom = len(custom_candidates) > 0

            annotation = {
                'idx': current_data['idx'],
                'sentence': current_data['sen'],
                'prd_word': current_data['prd_word'],
                'prd_idx': current_data['prd_idx'],
                'label': current_data['label'],
                'span_mean': current_data['span_mean'],
                'type': current_data.get('type', ''),
                'optional': is_optional,
                'timestamp': pd.Timestamp.now().isoformat(),
                'grammar_status': grammar_status,
                'grammar_error_desc': error_description if grammar_status == "Has grammatical errors" else "",
                'selected_spans': final_spans,
                'has_custom': has_custom,
            }

            # Update or insert
            existing_idx = next(
                (i for i, ann in enumerate(st.session_state.annotations)
                 if ann['idx'] == current_data['idx']),
                None
            )
            if existing_idx is not None:
                st.session_state.annotations[existing_idx] = annotation
                st.toast("✅ Annotation updated", icon="✅")
            else:
                st.session_state.annotations.append(annotation)
                st.toast("✅ Annotation saved", icon="✅")

            save_annotations()

            if st.session_state.current_idx < len(st.session_state.data) - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.balloons()
                st.info("🎉 This is the last sentence")

        if skip_button:
            if st.session_state.current_idx < len(st.session_state.data) - 1:
                st.session_state.current_idx += 1
                st.rerun()
            else:
                st.info("🎉 This is the last sentence")

    # ── Navigation Buttons ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("◀ Previous", disabled=st.session_state.current_idx == 0, use_container_width=True):
            st.session_state.current_idx -= 1
            st.rerun()
    with col2:
        if st.button("Next ▶", disabled=st.session_state.current_idx >= len(st.session_state.data) - 1, use_container_width=True):
            st.session_state.current_idx += 1
            st.rerun()
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col4:
        if st.button("📊 Results", use_container_width=True, type="secondary"):
            st.session_state.show_results = not st.session_state.get('show_results', False)
            st.rerun()

    # ── Annotation Results Display ────────────────────────────────────────
    if st.session_state.get('show_results', False) and st.session_state.annotations:
        st.markdown("**📋 Annotation Results**")

        df_data = []
        for ann in st.session_state.annotations:
            spans = ann.get('selected_spans', [])
            if isinstance(spans, list) and len(spans) == 0:
                span_text = 'No Argument'
                span_start = span_end = source = ''
            else:
                span_text = ' | '.join([s['text'] for s in spans])
                span_start = ' | '.join([str(s['start']) for s in spans])
                span_end = ' | '.join([str(s['end']) for s in spans])
                source = ' | '.join([','.join(s.get('models', [])) for s in spans])

            df_data.append({
                'Index': ann['idx'],
                'Predicate': ann['prd_word'],
                'Label': ann['label'],
                'Argument Meaning': ann['span_mean'],
                'Selected Span': span_text,
                'Start': span_start,
                'End': span_end,
                'Source': source,
                'Optional': '⚠️' if ann.get('optional') else '',
                'Grammar Status': ann.get('grammar_status', 'N/A'),
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=300)

        # Summary statistics
        total_ann = len(df_data)
        optional_ann = sum(1 for a in st.session_state.annotations if a.get('optional'))
        no_arg_ann = sum(1 for a in st.session_state.annotations if isinstance(a.get('selected_spans'), list) and len(a['selected_spans']) == 0)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("✅ Annotated", total_ann)
        with col_s2:
            st.metric("⚠️ Optional", optional_ann)
        with col_s3:
            st.metric("🚫 No Argument", no_arg_ann)

        # Export
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export JSON", use_container_width=True):
                st.json(st.session_state.annotations)
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name=f"annotations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()