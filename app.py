import streamlit as st
import fitz  # PyMuPDF
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

# --- [핵심 수정] 문항과 선지를 하나로 묶는 추출 함수 ---
def extract_problems_with_pages(file):
    if file is None: return []
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_problems = []
    
    noise_keywords = ['학년도', '영역', '확인사항', '유의사항', '성명', '수험번호', '생활과 윤리']

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        
        # 줄바꿈 단위로 먼저 쪼갬
        lines = page_text.split('\n')
        
        current_prob = ""
        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line: continue
            
            # 새 문항의 시작 패턴 (숫자. 또는 [숫자] 또는 숫자))
            is_new_start = bool(re.match(r'^(\d+[\.|\)]|\[\d+\])', cleaned_line))
            
            # 만약 새 번호로 시작하고, 기존에 쌓인 문장이 있다면 저장
            if is_new_start and current_prob:
                # 노이즈 필터링 후 저장
                if len(current_prob) >= 45 and not any(nk in current_prob[:30] for nk in noise_keywords):
                    all_problems.append({"text": current_prob, "page": page_num + 1})
                current_prob = cleaned_line # 새 문항 시작
            else:
                # 번호로 시작하지 않는 선지나 본문 내용은 이전 내용에 합침
                if current_prob:
                    current_prob += " " + cleaned_line
                else:
                    # 문서 맨 처음 시작 처리
                    current_prob = cleaned_line

        # 마지막 문항 처리
        if current_prob and len(current_prob) >= 45:
            all_problems.append({"text": current_prob, "page": page_num + 1})
                
    return all_problems

# --- [변별력 하이라이트] 6글자 이상 일치 시 ---
def highlight_selective(target, reference):
    ref_stripped = re.sub(r'\s+', '', reference)
    min_match_len = 6 
    
    to_highlight = []
    for i in range(len(target) - min_match_len + 1):
        chunk = target[i:i+min_match_len]
        if len(chunk.strip()) < min_match_len: continue
        
        chunk_stripped = re.sub(r'\s+', '', chunk)
        if chunk_stripped in ref_stripped:
            to_highlight.append(chunk)

    sorted_chunks = sorted(list(set(to_highlight)), key=len, reverse=True)
    result = target
    for chunk in sorted_chunks:
        if chunk in result:
            result = result.replace(chunk, f"[[M_S]]{chunk}[[M_E]]")
    
    result = result.replace("[[M_S]]", "<mark>").replace("[[M_E]]", "</mark>")
    return re.sub(r'</mark><mark>', '', result)

# --- UI 레이아웃 ---
st.title("🔍 문항 유사도 정밀 분석기")

col1, col2 = st.columns(2)
with col1:
    file_origin = st.file_uploader("📘 기준 PDF (수특/평가원)", type="pdf", key="origin")
with col2:
    file_new = st.file_uploader("📝 대상 PDF (출제 문항)", type="pdf", key="new")

if file_origin and file_new:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✨ 분석 시작하기"):
        with st.spinner('문항과 선지를 통합하여 분석 중입니다...'):
            list_origin = extract_problems_with_pages(file_origin)
            list_new = extract_problems_with_pages(file_new)
            
            if list_origin and list_new:
                results = []
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
    st.markdown("---")
    for res in st.session_state.results:
        status = "✅"
        if res['score'] > 65: status = "🚨 위험"
        elif res['score'] > 35: status = "⚠️ 주의"
        
        label = f"{status} | {res['id']}번 문항 (유사도 {res['score']}%)[원본 {res['page']}p]"
        with st.expander(label):
            h_new = highlight_selective(res['new'], res['origin'])
            h_origin = highlight_selective(res['origin'], res['new'])
            
            c1, c2 = st.columns(2)
            with c1: st.markdown(f"<div class='compare-box'><b>[출제 문항]</b><hr>{h_new}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='compare-box'><b>[기준 문항 - {res['page']}p]</b><hr>{h_origin}</div>", unsafe_allow_html=True)
