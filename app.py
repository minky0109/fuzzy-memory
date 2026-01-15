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
        background-color: white; color: black; min-height: 250px; line-height: 1.8;
        font-size: 1.05rem;
    }
    mark { background-color: #FFD1DC; color: #D63384; font-weight: bold; border-radius: 3px; padding: 0 1px; }
    </style>
    """, unsafe_allow_html=True)

# --- [개선] 노이즈 제거 및 텍스트 정규화 ---
def clean_text(text):
    # 줄바꿈, 탭, 여러 개의 공백을 하나의 공백으로 통일
    text = re.sub(r'\s+', ' ', text)
    # 특수 기호 정규화 (비교 정확도 향상)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    return text.strip()

def extract_problems_with_pages(file):
    if file is None: return []
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    
    noise_words = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '생활과 윤리']

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        
        # 문항 번호 패턴 쪼개기
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        
        for p in split_text:
            cleaned_p = clean_text(p)
            
            # [필터] 숫자로 시작 안하거나 너무 짧거나 노이즈 단어가 앞부분에 있으면 통과
            if not re.match(r'^(\d+|\[\d+|[①-⑳])', cleaned_p) or len(cleaned_p) < 45:
                continue
            if any(nw in cleaned_p[:30] for nw in noise_words):
                continue

            all_problems.append({"text": cleaned_p, "page": page_num + 1})
    return all_problems

# --- [핵심 보완] 슬라이딩 윈도우 기반 하이라이트 ---
def highlight_precise(target, reference):
    """
    단어 단위가 아니라 4글자 이상의 공통 문자열을 찾아 하이라이트합니다.
    조사나 어미가 달라도 핵심 문구는 모두 잡아냅니다.
    """
    # 비교를 위해 공백 제거 버전 생성
    ref_stripped = re.sub(r'\s+', '', reference)
    
    # 공백을 포함한 원문에서 4글자 이상의 공통 부분 찾기
    # 최소 4글자 연속 일치 시 하이라이트 대상
    min_len = 4
    to_highlight = set()
    
    # 타겟 텍스트에서 윈도우를 밀면서 참조 텍스트에 존재하는지 확인
    words = target.split()
    for i in range(len(target) - min_len + 1):
        chunk = target[i:i+min_len]
        if chunk.strip() == "": continue
        
        # 공백 제거하고 비교 (조사 차이 극복)
        chunk_stripped = re.sub(r'\s+', '', chunk)
        if chunk_stripped in ref_stripped and len(chunk_stripped) >= 3:
            to_highlight.add(chunk)

    # 하이라이트할 문구들을 길이 순(긴 것부터) 정렬
    sorted_chunks = sorted(list(to_highlight), key=len, reverse=True)
    
    result = target
    for chunk in sorted_chunks:
        # 중복 하이라이트 방지를 위해 간단한 치환 사용
        if chunk in result:
            result = result.replace(chunk, f"<mark>{chunk}</mark>")
    
    # 중첩된 mark 태그 정리 (정규식 사용)
    result = re.sub(r'</mark><mark>', '', result)
    return result

# --- UI 및 분석 로직 ---
st.title("🔍 문항 유사도 정밀 분석기 (정확도 강화)")

col1, col2 = st.columns(2)
with col1:
    file_origin = st.file_uploader("📘 기준 PDF", type="pdf")
with col2:
    file_new = st.file_uploader("📝 대상 PDF", type="pdf")

if file_origin and file_new:
    if st.button("✨ 분석 시작하기"):
        with st.spinner('문자 단위로 정밀 대조 중입니다. 잠시만 기다려주세요...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            if not list_origin or not list_new:
                st.error("문항을 찾지 못했습니다.")
            else:
                results = []
                # 유사도 분석은 문장 흐름(n-gram) 반영
                vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char') 
                
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
    st.markdown("---")
    for res in st.session_state.results:
        status = "✅"
        if res['score'] > 40:
            status = "🚨 위험" if res['score'] > 70 else "⚠️ 주의"
        
        label = f"{status} | {res['id']}번 (유사도 {res['score']}%)[원본 {res['page']}p]"
        with st.expander(label):
            # 정밀 하이라이트 적용
            h_new = highlight_precise(res['new'], res['origin'])
            h_origin = highlight_precise(res['origin'], res['new'])
            
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"<div class='compare-box'><b>[출제 문항]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='compare-box'><b>[기준 문항 - {res['page']}p]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
