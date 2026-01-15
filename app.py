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

# --- [개선] 문항 번호와 페이지를 동시에 추출 ---
def extract_problems_with_details(file):
    if file is None: return []
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    noise_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호']

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        lines = page.get_text("text").split('\n')
        
        current_prob = ""
        current_num = "" # 현재 문항의 번호 저장

        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line: continue
            
            # 문제 번호 패턴 매칭 (1. 또는 [01] 또는 1))
            match = re.match(r'^(\d+[\.|\)]|\[\d+\])', cleaned_line)
            
            if match and current_prob:
                # 이전까지 쌓인 문항 저장
                if len(current_prob) >= 40 and not any(nk in current_prob[:30] for nk in noise_keywords):
                    all_problems.append({
                        "text": current_prob, 
                        "page": page_num + 1,
                        "num": current_num
                    })
                # 새 문항 시작
                current_num = match.group(1).strip()
                current_prob = cleaned_line
            elif match:
                # 첫 문항 시작
                current_num = match.group(1).strip()
                current_prob = cleaned_line
            else:
                # 번호가 없으면 이전 문항에 합침
                current_prob = (current_prob + " " + cleaned_line) if current_prob else cleaned_line

        # 마지막 문항 처리
        if current_prob and len(current_prob) >= 40:
            all_problems.append({
                "text": current_prob, 
                "page": page_num + 1,
                "num": current_num
            })
    return all_problems

# --- 유사도 산출 로직 (기존 보정값 유지) ---
def calculate_custom_similarity(text1, text2):
    vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer='char')
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        v_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except: v_score = 0
    s1, s2 = re.sub(r'\s+', '', text1), re.sub(r'\s+', '', text2)
    common_len = sum(1 for i in range(len(s1)-5) if s1[i:i+5] in s2)
    ratio_score = (common_len * 1.5) / max(len(s1), 1)
    return min(round(((v_score * 0.4) + (ratio_score * 0.6)) * 100, 1), 100.0)

# --- 하이라이트 로직 ---
def highlight_selective(target, reference):
    ref_stripped = re.sub(r'\s+', '', reference)
    min_match_len = 6
    to_highlight = []
    for i in range(len(target)-min_match_len+1):
        chunk = target[i:i+min_match_len]
        if len(chunk.strip()) >= min_match_len and re.sub(r'\s+', '', chunk) in ref_stripped:
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
    file_origin = st.file_uploader("📘 기준 PDF (수특/평가원)", type="pdf", key="origin")
with col2:
    file_new = st.file_uploader("📝 대상 PDF (출제 문항)", type="pdf", key="new")

if file_origin and file_new:
    if st.button("✨ 분석 시작하기"):
        with st.spinner('문항 번호와 페이지를 매칭하여 대조 중입니다...'):
            list_origin = extract_problems_with_details(file_origin)
            list_new = extract_problems_with_details(file_new)
            
            if list_origin and list_new:
                results = []
                for i, new_item in enumerate(list_new):
                    best_score, best_match, found_page, found_num = 0, "", 0, ""
                    for origin_item in list_origin:
                        score = calculate_custom_similarity(new_item['text'], origin_item['text'])
                        if score > best_score:
                            best_score = score
                            best_match = origin_item['text']
                            found_page = origin_item['page']
                            found_num = origin_item['num'] # 원본 문항 번호 저장
                    
                    results.append({
                        "id": i + 1, "score": best_score,
                        "origin": best_match, "new": new_item['text'], 
                        "page": found_page, "origin_num": found_num
                    })
                st.session_state.results = results

if 'results' in st.session_state:
    st.markdown("---")
    for res in st.session_state.results:
        status = "✅"
        if res['score'] > 60: status = "🚨 위험"
        elif res['score'] > 30: status = "⚠️ 주의"
        
        # [수정] 제목에 페이지와 원본 문항 번호를 명시
        origin_info = f" [원본 {res['page']}p {res['origin_num']}]" if res['origin_num'] else f" [원본 {res['page']}p]"
        label = f"{status} | {res['id']}번 문항 (유사도 {res['score']}%){origin_info}"
        
        with st.expander(label):
            h_new = highlight_selective(res['new'], res['origin'])
            h_origin = highlight_selective(res['origin'], res['new'])
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"<div class='compare-box'><b>[출제 문항]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='compare-box'><b>[유사 문항 - {res['page']}p {res['origin_num']}]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
