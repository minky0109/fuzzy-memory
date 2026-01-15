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
    mark { background-color: #FFC0CB; color: black; font-weight: bold; border-radius: 3px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- [개선] 텍스트 추출 및 선지 통합 ---
def extract_problems_with_pages(file):
    if file is None: return []
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    noise_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '생활과 윤리']

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        lines = page.get_text("text").split('\n')
        current_prob = ""
        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line: continue
            # 문제 번호로 시작하는지 체크
            is_new_start = bool(re.match(r'^(\d+[\.|\)]|\[\d+\])', cleaned_line))
            if is_new_start and current_prob:
                if len(current_prob) >= 40 and not any(nk in current_prob[:30] for nk in noise_keywords):
                    all_problems.append({"text": current_prob, "page": page_num + 1})
                current_prob = cleaned_line
            else:
                current_prob = (current_prob + " " + cleaned_line) if current_prob else cleaned_line
        if current_prob and len(current_prob) >= 40:
            all_problems.append({"text": current_prob, "page": page_num + 1})
    return all_problems

# --- [개선] 퍼센테이지 도출 로직 (글자 단위 정밀 비교) ---
def calculate_custom_similarity(text1, text2):
    """
    단순 벡터 비교가 아니라, 두 문장에서 공통으로 발견되는 
    글자 뭉치(n-gram)의 비율을 계산하여 점수를 보정합니다.
    """
    # 1. 기본적인 벡터 유사도 (문맥 파악)
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        v_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        v_score = 0

    # 2. 실질적 중복 비율 계산 (글자 기반)
    s1 = re.sub(r'\s+', '', text1)
    s2 = re.sub(r'\s+', '', text2)
    
    # 더 짧은 쪽을 기준으로 얼마나 겹치는지 체크
    common_len = 0
    match_len = 5 # 5글자 이상 연속 일치 시 점수 가산
    for i in range(len(s1) - match_len + 1):
        if s1[i:i+match_len] in s2:
            common_len += 1
            
    # 벡터 점수와 실무적 겹침 점수를 혼합 (가중치 조정 가능)
    # 실제 문구가 많이 겹칠수록 점수가 정직하게 올라가도록 보정
    ratio_score = (common_len * 1.5) / max(len(s1), 1)
    final_score = (v_score * 0.4) + (ratio_score * 0.6)
    
    return min(round(final_score * 100, 1), 100.0)

# --- [동일] 하이라이트 로직 ---
def highlight_selective(target, reference):
    ref_stripped = re.sub(r'\s+', '', reference)
    min_match_len = 6 
    to_highlight = []
    for i in range(len(target) - min_match_len + 1):
        chunk = target[i:i+min_match_len]
        if len(chunk.strip()) < min_match_len: continue
        if re.sub(r'\s+', '', chunk) in ref_stripped:
            to_highlight.append(chunk)
    sorted_chunks = sorted(list(set(to_highlight)), key=len, reverse=True)
    result = target
    for chunk in sorted_chunks:
        if chunk in result: result = result.replace(chunk, f"[[MS]]{chunk}[[ME]]")
    result = result.replace("[[MS]]", "<mark>").replace("[[ME]]", "</mark>")
    return re.sub(r'</mark><mark>', '', result)

# --- UI 레이아웃 ---
st.title("🔍 문항 유사도 정밀 분석기")

col1, col2 = st.columns(2)
with col1:
    file_origin = st.file_uploader("📘 기준 PDF", type="pdf", key="origin")
with col2:
    file_new = st.file_uploader("📝 대상 PDF", type="pdf", key="new")

if file_origin and file_new:
    if st.button("✨ 분석 시작하기"):
        with st.spinner('문항별 유사도 점수를 정밀하게 계산 중입니다...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            if list_origin and list_new:
                results = []
                for i, new_item in enumerate(list_new):
                    best_score, best_match, found_page = 0, "", 0
                    for origin_item in list_origin:
                        score = calculate_custom_similarity(new_item['text'], origin_item['text'])
                        if score > best_score:
                            best_score, best_match, found_page = score, origin_item['text'], origin_item['page']
                    
                    results.append({
                        "id": i + 1, "score": best_score,
                        "origin": best_match, "new": new_item['text'], "page": found_page
                    })
                st.session_state.results = results

if 'results' in st.session_state:
    st.markdown("---")
    for res in st.session_state.results:
        status = "✅"
        if res['score'] > 60: status = "🚨 위험"
        elif res['score'] > 30: status = "⚠️ 주의"
        
        label = f"{status} | {res['id']}번 문항 (유사도 {res['score']}%)[원본 {res['page']}p]"
        with st.expander(label):
            h_new = highlight_selective(res['new'], res['origin'])
            h_origin = highlight_selective(res['origin'], res['new'])
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"<div class='compare-box'><b>[출제]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='compare-box'><b>[기준 - {res['page']}p]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
