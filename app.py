import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="문항 유사도 분석기", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFF5F7; }
    h1, h2, h3 { color: #D63384; }
    div.stButton > button {
        width: 100%; background-color: #FFB6C1; color: white;
        border-radius: 12px; border: none; height: 3.5em; font-weight: bold;
    }
    .compare-box {
        border: 2px solid #FFB6C1; padding: 20px; border-radius: 15px;
        background-color: white; color: black; min-height: 150px; line-height: 1.6;
    }
    mark { background-color: #FFD1DC; color: black; font-weight: bold; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

# --- 유틸리티 함수 정의 ---
def extract_problems_with_pages(file):
    if file is None: return []
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        for p in split_text:
            cleaned_p = p.strip()
            if len(cleaned_p) > 15:
                all_problems.append({"text": cleaned_p, "page": page_num + 1})
    return all_problems

def highlight_common_words(text, reference_text):
    ref_words = set(re.findall(r'\b\w{2,}\b', reference_text))
    target_words = re.findall(r'\b\w{2,}\b', text)
    highlighted_text = text
    for word in sorted(list(set(target_words)), key=len, reverse=True):
        if word in ref_words:
            # 특수문자 처리를 위해 re.escape 사용
            highlighted_text = re.sub(f'({re.escape(word)})', r'<mark>\1</mark>', highlighted_text)
    return highlighted_text

# --- UI 섹션 ---
st.title("🔍 문항 유사도 정밀 분석기")

# 변수 초기화 (NameError 방지)
file_origin = None
file_new = None

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📘 기준 PDF (수특/평가원)")
    file_origin = st.file_uploader("파일을 선택하세요", type="pdf", key="origin_upload")
with col2:
    st.markdown("#### 📝 대상 PDF (출제자)")
    file_new = st.file_uploader("파일을 선택하세요", type="pdf", key="new_upload")

# 2. 분석 실행 버튼
if file_origin is not None and file_new is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 분석 시작하기"):
        with st.spinner('페이지별 데이터를 정밀 분석 중입니다...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            results = []
            if list_origin and list_new:
                vectorizer = TfidfVectorizer()
                for i, new_item in enumerate(list_new):
                    new_p = new_item['text']
                    best_score, best_match, found_page = 0, "매칭 항목 없음", 0
                    
                    for origin_item in list_origin:
                        origin_p = origin_item['text']
                        try:
                            tfidf = vectorizer.fit_transform([new_p, origin_p])
                            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                            if score > best_score:
                                best_score, best_match, found_page = score, origin_p, origin_item['page']
                        except: continue
                    
                    results.append({
                        "id": i + 1, "score": round(best_score * 100, 1),
                        "origin": best_match, "new": new_p, "page": found_page
                    })
                st.session_state.results = results
            else:
                st.error("파일에서 문항을 추출하지 못했습니다. PDF 형식을 확인해주세요.")

# 3. 결과 표시
if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("📋 분석 결과")
    for res in st.session_state.results:
        status = "✅"
        page_info = ""
        if res['score'] > 40:
            status = "🚨 위험" if res['score'] > 70 else "⚠️ 주의"
            page_info = f" [원본 {res['page']}페이지]"
        
        label = f"{status} | {res['id']}번 문항 (유사도: {res['score']}%){page_info}"
        with st.expander(label):
            h_new = highlight_common_words(res['new'], res['origin'])
            h_origin = highlight_common_words(res['origin'], res['new'])
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"<div class='compare-box'><b>[출제 문항]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='compare-box'><b>[기준 문항 - {res['page']}p]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
