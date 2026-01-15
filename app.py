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
    
    # [중요] 파일 읽기 위치 초기화 (페이지 누락 방지 핵심)
    file.seek(0)
    
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    
    # [필터] 절대 문항이 될 수 없는 키워드 (여기에 '확인사항' 추가)
    exclude_keywords = [
        '수능특강', '발행처', 'EBS', '페이지', '과목', '학년도', 
        '모의평가', '시험지', '교재', '판권', '확인사항', '유의사항', 
        '정답과 해설', '수험번호', '성명'
    ]

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        current_page_no = page_num + 1
        
        # 1. 문항 번호(숫자+점, 숫자+괄호 등) 기준으로 쪼개기
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        
        for p in split_text:
            cleaned_p = p.strip()
            
            # [조건 1] 너무 짧은 텍스트는 무시 (45자 미만)
            if len(cleaned_p) < 45:
                continue
            
            # [조건 2] 제외 키워드 필터링 (특히 '확인사항' 차단)
            is_noise = False
            for key in exclude_keywords:
                if key in cleaned_p:
                    # 키워드가 포함되어 있는데, 숫자로 시작하지 않는다면 100% 노이즈(헤더/공지)
                    if not re.match(r'^\d', cleaned_p):
                        is_noise = True
                        break
            
            if not is_noise:
                all_problems.append({
                    "text": cleaned_p, 
                    "page": current_page_no  # 현재 분석 중인 실제 페이지 번호 기록
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
st.write("PDF의 '확인사항' 등 불필요한 정보는 제외하고 문항만 정밀하게 분석합니다.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📘 기준 PDF (수특/평가원)")
    file_origin = st.file_uploader("파일 선택", type="pdf", key="origin")
with col2:
    st.markdown("#### 📝 대상 PDF (출제 문항)")
    file_new = st.file_uploader("파일 선택", type="pdf", key="new")

# 분석 실행 버튼
if file_origin and file_new:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 분석 시작하기"):
        with st.spinner('페이지별 문항을 정밀하게 대조하는 중...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            if not list_origin or not list_new:
                st.error("파일에서 분석 가능한 문항을 찾지 못했습니다.")
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
        status = "✅"
        page_display = f"{res['page']}p" if res['page'] > 0 else "정보없음"
        page_tag = ""
        
        if res['score'] > 40:
            status = "🚨 위험" if res['score'] > 70 else "⚠️ 주의"
            page_tag = f" [원본 {page_display}]"
        
        label = f"{status} | {res['id']}번 문항 (유사도 {res['score']}%){page_tag}"
        
        with st.expander(label):
            h_new = highlight_common_words(res['new'], res['origin'])
            h_origin = highlight_common_words(res['origin'], res['new'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='compare-box'><b>[출제 문항]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='compare-box'><b>[기준 문항 - {page_display}]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
