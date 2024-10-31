import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import re

# .env 파일에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI Chat 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

# 웹페이지에서 수상작을 자동으로 추출하는 함수
def fetch_awards_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    main_content = soup.find("div", {"class": "blog-content"})  # 주요 콘텐츠를 포함한 div 찾기
    
    if not main_content:
        return "Error: The main content could not be found. Please check the HTML structure."

    # 각 수상작 구분 텍스트와 요약을 자동으로 찾기
    award_sections = {"대상": [], "우수상": [], "입선": []}
    current_award_type = None
    
    for element in main_content.find_all(["p", "h3", "h2"]):  # 예시로 p, h3, h2 태그를 사용하여 수상작 정보를 찾음
        text = element.get_text().strip()
        
        # 수상작 종류를 인식
        if "대상" in text:
            current_award_type = "대상"
        elif "우수상" in text:
            current_award_type = "우수상"
        elif "입선" in text:
            current_award_type = "입선"
        
        # 수상작 요약을 각 항목에 추가
        if current_award_type:
            award_sections[current_award_type].append(text)
    
    # 각 수상작별 요약을 정리
    summaries = {}
    for award, contents in award_sections.items():
        summaries[award] = " ".join(contents)

    return summaries

# 수상작 내용을 자동으로 추출
url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
award_summaries = fetch_awards_content(url)

# 로컬 Chroma 벡터 스토어 설정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(" ".join(award_summaries.values()))
documents = [Document(page_content=chunk) for chunk in chunks]
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = Chroma.from_documents(documents, embeddings, persist_directory="chroma_store")
vector_store.persist()

# Streamlit 설정 및 챗봇 UI
st.title("All-in Coding Challenge RAG Chatbot")
st.write("이번 챗봇은 'ALL-in 코딩 공모전' 수상작 정보를 요약해주는 RAG 시스템입니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("챗봇에게 질문을 입력하세요:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "수상작" in prompt or "대상" in prompt or "우수상" in prompt or "입선" in prompt:
        response = "다음은 'ALL-in 코딩 공모전' 수상작 요약입니다:\n\n"
        for award, summary in award_summaries.items():
            response += f"**{award}**: {summary}\n\n"
    else:
        docs = vector_store.similarity_search(prompt)
        question_template = PromptTemplate(input_variables=["question"], template="다음은 'ALL-in 코딩 공모전' 수상작 요약입니다. 질문에 대해 답변해주세요: {question}")
        formatted_prompt = question_template.format(question=prompt)
        
        answer = llm([HumanMessage(content=formatted_prompt)])
        response = answer.content

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with open("conversation_log.txt", "a") as f:
        f.write(f"사용자: {prompt}\n챗봇: {response}\n\n")
