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
        border-radius: 12px; border: none; height: 3.5em; font-weight: bold; font-size: 1.1rem;
    }
    div.stButton > button:hover { background-color: #FF8DA1; color: white; transform: translateY(-2px); }
    .compare-box {
        border: 2px solid #FFB6C1; padding: 20px; border-radius: 15px;
        background-color: white; color: black; min-height: 200px; line-height: 1.7;
    }
    mark { background-color: #FFD1DC; color: black; font-weight: bold; border-radius: 3px; padding: 0 2px; }
    .streamlit-expanderHeader { border: 1px solid #FFB6C1 !important; border-radius: 10px !important; background-color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 텍스트 추출 및 타이틀/노이즈 제거 함수 ---
def extract_problems_with_pages(file):
    if file is None: return []
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    
    # [강력 필터] 타이틀 및 안내문구 키워드
    noise_keywords = [
        '학년도', '영역', '생활과 윤리', '윤리와 사상', '사회·문화', '지리', '역사', 
        '정답과 해설', '확인사항', '유의사항', '수험번호', '성명', 'EBS', '수능특강'
    ]

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        current_page_no = page_num + 1
        
        # 문항 번호 패턴으로 쪼개기
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        
        for p in split_text:
            cleaned_p = p.strip()
            
            # [필터 1] 너무 짧은 건 무조건 패스
            if len(cleaned_p) < 45: continue
            
            # [필터 2] 숫자로 시작하는지 확인 (진짜 문항은 보통 1. 또는 [01]로 시작)
            starts_with_num = bool(re.match(r'^\d|^\[\d', cleaned_p))
            
            # [필터 3] 타이틀 노이즈 검사
            is_noise = False
            for key in noise_keywords:
                if key in cleaned_p:
                    # 숫자로 시작하지 않으면서 과목명이 들어있으면 100% 타이틀 노이즈
                    if not starts_with_num:
                        is_noise = True
                        break
            
            if not is_noise:
                all_problems.append({
                    "text": cleaned_p, 
                    "page": current_page_no
                })
                
    return all_problems

def highlight_common_words(text, reference_text):
    ref_words = set(re.findall(r'\b\w{2,}\b', reference_text))
    target_words = re.findall(r'\b\w{2,}\b', text)
    highlighted_text = text
    for word in sorted(list(set(target_words)), key=len, reverse=True):
        if word in ref_words:
            highlighted_text = re.sub(f'({re.escape(word)})', r'<mark>\1</mark>', highlighted_text)
    return highlighted_text

# --- UI 레이아웃 ---
st.title("🔍 문항 유사도 정밀 분석기")
st.write("시험지 타이틀(학년도, 과목명) 및 확인사항을 자동으로 제외하고 분석합니다.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📘 기준 PDF (수특/평가원)")
    file_origin = st.file_uploader("파일 선택", type="pdf", key="origin")
with col2:
    st.markdown("#### 📝 대상 PDF (출제 문항)")
    file_new = st.file_uploader("파일 선택", type="pdf", key="new")

if file_origin and file_new:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 분석 시작하기"):
        with st.spinner('타이틀을 제외하고 문항만 정밀 대조 중입니다...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            if not list_origin or not list_new:
                st.error("분석할 문항을 찾지 못했습니다. PDF 구성을 확인해주세요.")
            else:
                results = []
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

if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("📋 분석 리포트")
    for res in st.session_state.results:
        status = "✅"
        page_tag = ""
        if res['score'] > 40:
            status = "🚨 위험" if res['score'] > 70 else "⚠️ 주의"
            page_tag = f" [원본 {res['page']}p]"
        
        label = f"{status} | {res['id']}번 문항 (유사도 {res['score']}%){page_tag}"
        with st.expander(label):
            h_new = highlight_common_words(res['new'], res['origin'])
            h_origin = highlight_common_words(res['origin'], res['new'])
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"<div class='compare-box'><b>[출제 문항]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='compare-box'><b>[기준 문항 - {res['page']}p]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
