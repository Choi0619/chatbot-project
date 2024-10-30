import os
import streamlit as st
from langchain import LangChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

# .env 파일에서 API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI Chat 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

# URL에서 텍스트를 크롤링하고 필요한 정보만 추출
def fetch_blog_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    main_content = soup.find("div", {"class": "blog-content"})  # 필요한 HTML 태그 지정
    paragraphs = main_content.find_all("p")
    content = " ".join([p.get_text() for p in paragraphs])
    return content

# 블로그 URL에서 내용 추출 및 분할
url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
content = fetch_blog_content(url)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_text(content)

# 텍스트 데이터를 벡터 스토어에 저장
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = Chroma.from_documents(documents, embeddings)

# Streamlit 설정 및 챗봇 UI
st.title("All-in Coding Challenge RAG Chatbot")
st.write("이번 챗봇은 'ALL-in 코딩 공모전' 수상작 정보를 요약해주는 RAG 시스템입니다.")
user_input = st.text_input("챗봇에게 질문을 입력하세요:")

# RAG로 질문에 응답
if user_input:
    docs = vector_store.similarity_search(user_input)
    prompt = f"다음은 'ALL-in 코딩 공모전' 수상작 요약입니다. {user_input} 질문에 대해 답변해주세요."
    answer = llm(prompt)  
    st.write("**챗봇의 답변:**", answer)
    
    # 대화 기록 저장
    with open("conversation_log.txt", "a") as f:
        f.write(f"사용자: {user_input}\n챗봇: {answer}\n\n")
