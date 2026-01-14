import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- [이전의 CSS 스타일 설정 부분은 동일하게 유지] ---

# 1. 페이지별로 텍스트를 추출하고 문항을 분리하는 함수
def extract_problems_with_pages(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        # 문제 번호 패턴으로 쪼개기
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        
        for p in split_text:
            cleaned_p = p.strip()
            if len(cleaned_p) > 15: # 너무 짧은 텍스트 제외
                all_problems.append({
                    "text": cleaned_p,
                    "page": page_num + 1  # 1페이지부터 시작하도록 +1
                })
    return all_problems

# --- [중간 하이라이트 함수 등은 동일하게 유지] ---

# 2. 분석 실행 로직 (수정됨)
if file_origin and file_new:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 분석 시작하기"):
        with st.spinner('페이지별 데이터를 정밀 분석 중입니다...'):
            # 기준 파일과 대상 파일 분석 (페이지 정보 포함)
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            results = []
            vectorizer = TfidfVectorizer()
            
            for i, new_item in enumerate(list_new):
                new_p = new_item['text']
                best_score = 0
                best_match = "매칭되는 문항 없음"
                found_page = 0
                
                for origin_item in list_origin:
                    origin_p = origin_item['text']
                    try:
                        tfidf = vectorizer.fit_transform([new_p, origin_p])
                        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                        if score > best_score:
                            best_score = score
                            best_match = origin_p
                            found_page = origin_item['page'] # 해당 문항의 원본 페이지 저장
                    except: continue
                
                results.append({
                    "id": i + 1,
                    "score": round(best_score * 100, 1),
                    "origin": best_match,
                    "new": new_p,
                    "page_info": found_page
                })
            st.session_state.results = results

# 3. 결과 출력 부분 (페이지 정보 노출 추가)
if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("📋 문항별 분석 결과")

    for res in st.session_state.results:
        status_icon = "✅"
        page_msg = ""
        
        # '주의' 이상의 유사도(40% 초과)일 때 페이지 정보 생성
        if res['score'] > 70:
            status_icon = "🚨 위험"
            page_msg = f"📍 [원본 PDF {res['page_info']}페이지 근처에서 발견]"
        elif res['score'] > 40:
            status_icon = "⚠️ 주의"
            page_msg = f"📍 [원본 PDF {res['page_info']}페이지 근처에서 발견]"
        
        label = f"{status_icon} | {res['id']}번 문항 (유사도: {res['score']}%) {page_msg}"
        
        with st.expander(label):
            # [기존과 동일한 상세 비교 레이아웃]
            h_new = highlight_common_words(res['new'], res['origin'])
            h_origin = highlight_common_words(res['origin'], res['new'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='compare-box'><b>[출제 문항]</b><br><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2:
                # 여기에 한 번 더 페이지 정보 강조
                st.markdown(f"<div class='compare-box'><b>[기준 문항 - {res['page_info']}페이지]</b><br><hr>{h_origin}</div>", unsafe_allow_html=True)
