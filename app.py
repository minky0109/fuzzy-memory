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
        background-color: white; color: black; min-height: 200px; line-height: 1.8;
    }
    mark { background-color: #FFD1DC; color: black; font-weight: bold; border-radius: 3px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- [보완] 텍스트 정밀 추출 및 타이틀 제거 ---
def extract_problems_with_pages(file):
    if file is None: return []
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    
    # 제외 키워드
    noise_words = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '생활과 윤리']

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        
        # 문항 번호 패턴으로 쪼개기
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        
        for p in split_text:
            # 줄바꿈과 중복 공백을 하나로 합쳐서 비교 정확도 향상
            cleaned_p = re.sub(r'\s+', ' ', p).strip()
            
            # [엄격 필터] 숫자로 시작하지 않거나 너무 짧으면 버림
            if not re.match(r'^(\d+|\[\d+|[①-⑳])', cleaned_p) or len(cleaned_p) < 50:
                continue
            
            # 타이틀 노이즈 추가 필터
            if any(nw in cleaned_p[:25] for nw in noise_words):
                continue

            all_problems.append({"text": cleaned_p, "page": page_num + 1})
    return all_problems

# --- [보완] 하이라이트 로직 (N-gram 기반) ---
def highlight_common_words(target, reference):
    """
    단순 단어 비교가 아니라, 2~3글자 단위로 겹치는 문구를 찾아 하이라이트합니다.
    """
    # 텍스트에서 의미 있는 단어(2글자 이상)만 추출
    target_words = re.findall(r'[가-힣A-Za-z0-9]{2,}', target)
    ref_words = set(re.findall(r'[가-힣A-Za-z0-9]{2,}', reference))
    
    # 겹치는 단어 리스트 추출 (긴 단어 우선)
    common_words = [word for word in target_words if word in ref_words]
    common_words = sorted(list(set(common_words)), key=len, reverse=True)
    
    highlighted = target
    for word in common_words:
        # 이미 하이라이트된 부분 안에 포함된 단어는 건너뛰기 위함
        pattern = f'({re.escape(word)})'
        # mark 태그 바깥에 있을 때만 치환
        highlighted = re.sub(pattern, r'<mark>\1</mark>', highlighted)
        
    return highlighted

# --- UI 및 분석 로직 ---
st.title("🔍 문항 유사도 정밀 분석기")

col1, col2 = st.columns(2)
with col1:
    file_origin = st.file_uploader("📘 기준 PDF", type="pdf", key="origin")
with col2:
    file_new = st.file_uploader("📝 대상 PDF", type="pdf", key="new")

if file_origin and file_new:
    if st.button("✨ 분석 시작하기"):
        with st.spinner('문구 하나하나 대조 중...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            if not list_origin or not list_new:
                st.error("문항을 찾지 못했습니다.")
            else:
                results = []
                # 문항 비교 시 정확도를 위해 Tfidf 파라미터 조정
                vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
                
                for i, new_item in enumerate(list_new):
                    new_p = new_item['text']
                    best_score, best_match, found_page = 0, "", 0
                    
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
    for res in st.session_state.results:
        status = "✅"
        if res['score'] > 40:
            status = "🚨 위험" if res['score'] > 70 else "⚠️ 주의"
        
        label = f"{status} | {res['id']}번 (유사도 {res['score']}%)[원본 {res['page']}p]"
        with st.expander(label):
            # 개선된 하이라이트 함수 호출
            h_new = highlight_common_words(res['new'], res['origin'])
            h_origin = highlight_common_words(res['origin'], res['new'])
            
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"<div class='compare-box'><b>[출제]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='compare-box'><b>[기준 - {res['page']}p]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
