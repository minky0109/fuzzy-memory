import streamlit as st
import fitz
import re
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
    }
    /* 하이라이트 색상을 조금 더 선명하게, 글자색은 검정 유지 */
    mark { background-color: #FFC0CB; color: black; font-weight: bold; border-radius: 3px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- [개선] 텍스트 정규화 (불필요한 공백 제거) ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_problems_with_pages(file):
    if file is None: return []
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    # 확실히 걸러야 할 노이즈 패턴
    noise_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호']

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        split_text = re.split(r'\n(?=\d+[\.|\)])|(?<=\n)(?=\d+[\.|\)])|(?=\[\d+\])', page_text)
        
        for p in split_text:
            cleaned_p = clean_text(p)
            # 숫자로 시작하고 일정 길이 이상인 것만
            if re.match(r'^(\d+|\[\d+|[①-⑳])', cleaned_p) and len(cleaned_p) >= 50:
                if not any(nk in cleaned_p[:30] for nk in noise_keywords):
                    all_problems.append({"text": cleaned_p, "page": page_num + 1})
    return all_problems

# --- [개선] 변별력 있는 하이라이트 (최소 6글자 일치 시에만) ---
def highlight_selective(target, reference):
    """
    흔한 단어는 무시하고, 6글자 이상의 고유한 문구가 겹칠 때만 하이라이트합니다.
    """
    ref_stripped = re.sub(r'\s+', '', reference)
    # 의미 없는 짧은 연결어들 (조사, 접속사 등 방지)
    # 최소 길이를 6글자로 상향하여 '변별력' 확보
    min_match_len = 6
    
    # 겹치는 구간 찾기
    to_highlight = []
    for i in range(len(target) - min_match_len + 1):
        chunk = target[i:i+min_match_len]
        if " " in chunk and len(chunk.strip()) < min_match_len: continue # 공백 제외 실질 글자수 체크
        
        chunk_stripped = re.sub(r'\s+', '', chunk)
        if chunk_stripped in ref_stripped:
            to_highlight.append(chunk)

    # 긴 문구부터 순차적으로 마킹 (중복 방지)
    sorted_chunks = sorted(list(set(to_highlight)), key=len, reverse=True)
    
    result = target
    for chunk in sorted_chunks:
        # 이미 mark 태그가 적용된 부분은 건드리지 않도록 보호
        if chunk in result:
            result = result.replace(chunk, f"[[MARK_START]]{chunk}[[MARK_END]]")
    
    result = result.replace("[[MARK_START]]", "<mark>").replace("[[MARK_END]]", "</mark>")
    # 연속된 mark 태그 병합
    result = re.sub(r'</mark><mark>', '', result)
    return result

# --- UI 및 분석 로직 ---
st.title("🔍 문항 유사도 정밀 분석기")

col1, col2 = st.columns(2)
with col1:
    file_origin = st.file_uploader("📘
