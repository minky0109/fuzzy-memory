import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정 및 디자인 (Pink 테마 유지)
st.set_page_config(page_title="문항 유사도 분석기", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFF5F7; }
    h1, h2, h3 { color: #D63384; }
    /* 분석 시작 버튼 스타일 */
    div.stButton > button {
        width: 100%;
        background-color: #FFB6C1;
        color: white;
        border-radius: 12px;
        border: none;
        height: 3.5em;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover { 
        background-color: #FF8DA1; 
        color: white; 
        transform: translateY(-2px);
        transition: 0.2s;
    }
    /* 문항 상세 박스 스타일 */
    .compare-box {
        border: 2px solid #FFB6C1;
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        color: black;
        min-height: 150px;
        margin-bottom: 10px;
        line-height: 1.6;
    }
    /* 하이라이트 효과 */
    mark { 
        background-color: #FFD1DC; 
        color: black; 
        font-weight: bold; 
        padding: 0 2px;
        border-radius: 3px;
    }
    /* Expander(버튼형 리스트) 스타일 */
    .streamlit-expanderHeader {
        background-color: white !important;
        border-radius: 10px !important;
        border: 1px solid #FFB6C1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 문항 유사도 정밀 분석기")
st.write("수평/평가원 대비 출제 문항의 중복 여부를 정밀하게 검사합니다.")

# 텍스트 처리 함수
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def get_problems(text):
    # 번호 패턴 추출 (1., 2., [1번] 등)
    problems = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', text)
    return [p.strip() for p in problems if len(p.strip()) > 15]

def highlight_common_words(text, reference_text):
    ref_words = set(re.findall(r'\b\w{2,}\b', reference_text))
    target_words = re.findall(r'\b\w{2,}\b', text)
    highlighted_text = text
    # 중복 단어 강조
    for word in sorted(list(set(target_words)), key=len, reverse=True):
        if word in ref_words:
            highlighted_text = re.sub(f'({re.escape(word)})', r'<mark>\1</mark>', highlighted_text)
    return highlighted_text

# 2. 파일 업로드 영역
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📘 기준 PDF (수특/평가원)")
    file_origin = st.file_uploader("파일을 선택하세요", type="pdf", key="origin")
with col2:
    st.markdown("#### 📝 대상 PDF (출제자)")
    file_new = st.file_uploader("파일을 선택하세요", type="pdf", key="new")

# 3. 분석 실행 로직
if file_origin and file_new:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 분석 시작하기"):
        with st.spinner('문항 데이터를 분석 중입니다. 잠시만 기다려주세요...'):
            text_origin = extract_text(file_origin)
            text_new = extract_text(file_new)
            
            list_origin = get_problems(text_origin)
            list_new = get_problems(text_new)
            
            results = []
            vectorizer = TfidfVectorizer()
            
            for i, new_p in enumerate(list_new):
                best_score = 0
                best_match = "매칭되는 문항을 찾을 수 없습니다."
                for origin_p in list_origin:
                    try:
                        tfidf = vectorizer.fit_transform([new_p, origin_p])
                        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                        if score > best_score:
                            best_score, best_match = score, origin_p
                    except: continue
                
                results.append({
                    "id": i + 1,
                    "score": round(best_score * 100, 1),
                    "origin": best_match,
                    "new": new_p
                })
            st.session_state.results = results

# 4. 결과 출력 (버튼형 리스트)
if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("📋 문항별 분석 결과")
    st.info("아래 문항 번호를 클릭하면 상세 비교 내용을 확인할 수 있습니다.")

    for res in st.session_state.results:
        # 유사도에 따른 라벨 설정
        status_icon = "✅"
        if res['score'] > 70: status_icon = "🚨 위험"
        elif res['score'] > 40: status_icon = "⚠️ 주의"
        
        label = f"{status_icon} | {res['id']}번 문항 (유사도: {res['score']}%)"
        
        # 버튼 형태의 상세 보기 (Expander)
        with st.expander(label):
            h_new = highlight_common_words(res['new'], res['origin'])
            h_origin = highlight_common_words(res['origin'], res['new'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='compare-box'><b>[출제 문항 내용]</b><br><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='compare-box'><b>[기준 문항 내용]</b><br><hr>{h_origin}</div>", unsafe_allow_html=True)