import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 0. JSON 데이터 로드 함수
@st.cache_data
def load_tifu_json(path: str) -> pd.DataFrame:
    records = []
    for fname in os.listdir(path):
        if fname.endswith('.json'):
            with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    records.append({
                        'text': f"Title: {data.get('title','')}\n{data.get('selftext','')}",
                        'source': 'TIFU',
                        'ups': data.get('ups', 0)
                    })
    return pd.DataFrame(records)

@st.cache_data
def load_aita_json(path: str) -> pd.DataFrame:
    records = []
    for fname in os.listdir(path):
        if fname.endswith('.json'):
            with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    verdict = data.get('judge', '')
                    records.append({
                        'text': f"Post: {data.get('text','')}\nVerdict: {verdict}",
                        'source': 'AITA',
                        'verdict': verdict
                    })
    return pd.DataFrame(records)

# 1. 벡터스토어 구축
@st.cache_resource
def build_vectorstore(tifu_path, aita_path, embedding_model="sentence-transformers/all-MiniLM-L6-v2", index_dir="./vectorstore"):
    df = pd.concat([load_tifu_json(tifu_path), load_aita_json(aita_path)], ignore_index=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    texts = df['text'].tolist()
    vectors = embeddings.embed_documents(texts)
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype='float32'))
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, 'index.faiss'))
    df.to_pickle(os.path.join(index_dir, 'metadata.pkl'))
    return index_dir

# 2. FAISS 로드
@st.cache_resource
def load_vectorstore(index_dir, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    # allow_dangerous_deserialization=True 로 pickle 로드 허용
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

# 3. QA 체인
@st.cache_resource
def init_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k":5})
    openai_key = st.secrets["openai"]["api_key"]
    llm = OpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=openai_key,
        temperature=0.7
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

# 4. Streamlit UI

def main():
    st.set_page_config(
        page_title="실수 리프레이머 챗봇",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 앱 헤더
    col1, col2 = st.columns([1,4])
    with col1:
        st.image("https://i.imgur.com/your_logo.png", width=80)
    with col2:
        st.title("😅 실수 리프레이머 챗봇")
        st.markdown("_작은 실수도 위인들의 실패와 비교하며 마음의 짐을 덜어보세요_ ")

    # 탭 구성
    tabs = st.tabs(["챗", "설정", "히스토리"])

    # 챗 탭
    with tabs[0]:
        st.subheader("여기에 실수를 입력해보세요!")
        user_input = st.text_area("당신의 실수:", height=100)
        if st.button("위로받기", key="run"):  
            if user_input.strip():
                with st.spinner("위로를 준비중입니다..."):
                    response = qa_chain.run(user_input)
                st.markdown("---")
                st.markdown("**🤖 리프레이밍 답변:**")
                st.write(response)
            else:
                st.warning("먼저 실수를 입력해주세요.")

    # 설정 탭
    with tabs[1]:
        st.subheader("데이터 & 인덱스 설정")
        tifu_path = st.text_input("TIFU JSON 폴더 경로", "./data/tifu_json")
        aita_path = st.text_input("AITA JSON 폴더 경로", "./data/aita_json")
        index_dir = st.text_input("FAISS 인덱스 폴더", "./vectorstore")
        if st.button("색인 재생성", key="reindex"):
            build_vectorstore(tifu_path, aita_path, index_dir=index_dir)
            st.success("색인이 재생성되었습니다.")
        st.caption("데이터 변경 시 재생성하세요.")

    # 히스토리 탭
    with tabs[2]:
        st.subheader("대화 히스토리")
        if 'history' not in st.session_state:
            st.session_state.history = []
        for entry in st.session_state.history:
            st.markdown(f"- **질문:** {entry['q']}")
            st.markdown(f"  **답변:** {entry['a']}\n")

if __name__ == '__main__':
    # 기본 경로 설정
    default_tifu = "./data/tifu_json"
    default_aita = "./data/aita_json"
    default_index = "./vectorstore"

    # 색인 파일 존재 여부 확인
    index_file = os.path.join(default_index, 'index.faiss')
    if not os.path.exists(index_file):
        st.warning("vectorstore가 존재하지 않습니다. 데이터 폴더를 불러와 색인을 생성합니다...")
        build_vectorstore(default_tifu, default_aita, index_dir=default_index)

    # 벡터스토어 로드 및 QA 체인 초기화
    vectorstore = load_vectorstore(default_index)
    qa_chain = init_qa_chain(vectorstore)

    main()
