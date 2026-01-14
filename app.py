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

# --- 텍스트 추출 및 정밀 필터링 함수 ---
def extract_problems_with_pages(file):
    if file is None: return []
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    
    # [필터] 문항이 아닌 텍스트에 자주 포함되는 단어들
    exclude_keywords = ['수능특강', '발행처', 'EBS', '페이지', '과목', '학년도', '모의평가', '시험지', '교재', '판권']

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        
        # 1. 문항 번호(1., 2., [01]) 기준으로 쪼개기
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        
        for p in split_text:
            cleaned_p = p.strip()
            
            # [조건 1] 너무 짧은 텍스트(헤더, 페이지번호 등)는 무시 (45자 기준)
            if len(cleaned_p) < 45:
                continue
            
            # [조건 2] 숫자로 시작하지 않으면서 제외 키워드가 포함된 경우 무시 (헤더 방지)
            is_header = False
            if not re.match(r'^\d', cleaned_p): # 숫자로 시작하지 않는데
                for key in exclude_keywords:
                    if key in cleaned_p:
                        is_header = True
                        break
            
            if not is_header:
                all_problems.append({"text": cleaned_p, "page": page_num + 1})
    return all_problems

def highlight_common_words(text, reference_text):
    # 조사/어미를 제외한 2글자 이상 단어 추출
    ref_words = set(re.findall(r'\b\w{2,}\b', reference_text))
    target_words = re.findall(r'\b\w{2,}\b', text)
    highlighted_text = text
    # 긴 단어부터 교체해야 짧은 단어 교체 시 꼬이지 않음
    for word in sorted(list(set(target_words)), key=len, reverse=True):
        if word in ref_words:
            highlighted_text = re.sub(f'({re.escape(word)})', r'<mark>\1</mark>', highlighted_text)
    return highlighted_text

# --- UI 레이아웃 ---
st.title("🔍 문항 유사도 정밀 분석기")
st.write("PDF 파일을 업로드하고 버튼을 누르면 문항별 유사도와 위치를 분석합니다.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📘 기준 PDF (수특/평가원)")
    file_origin = st.file_uploader("파일 업로드", type="pdf", key="origin")
with col2:
    st.markdown("#### 📝 대상 PDF (출제 문항)")
    file_new = st.file_uploader("파일 업로드", type="pdf", key="new")

# 분석 실행 버튼
if file_origin and file_new:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 분석 시작하기"):
        with st.spinner('문항을 추출하고 유사도를 비교하는 중입니다...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            if not list_origin or not list_new:
                st.error("문항을 제대로 읽어오지 못했습니다. PDF 내용을 확인해주세요.")
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

# 결과 섹션
if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("📋 분석 리포트")
    
    for res in st.session_state.results:
        # 상태 및 페이지 정보 설정
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
            with c1:
                st.markdown(f"<div class='compare-box'><b>[출제 문항 내용]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='compare-box'><b>[유사 문항 - {res['page']}페이지]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
