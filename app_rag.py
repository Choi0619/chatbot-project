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

    # main_content가 존재하지 않을 경우 처리
    if main_content:
        paragraphs = main_content.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
    else:
        content = "Error: The main content could not be found. Please check the HTML structure."
    
    return content

# 블로그 URL에서 내용 추출 및 분할
url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
content = fetch_blog_content(url)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(content)

# 텍스트 데이터를 Document 객체로 변환
documents = [Document(page_content=chunk) for chunk in chunks]

# 로컬 디렉토리에 저장되는 Chroma 벡터 스토어 설정
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = Chroma.from_documents(documents, embeddings, persist_directory="chroma_store")
vector_store.persist()  # 데이터 저장

# Streamlit 설정 및 챗봇 UI
st.title("All-in Coding Challenge RAG Chatbot")
st.write("이번 챗봇은 'ALL-in 코딩 공모전' 수상작 정보를 요약해주는 RAG 시스템입니다.")
user_input = st.text_input("챗봇에게 질문을 입력하세요:")

# RAG로 질문에 응답
if user_input:
    docs = vector_store.similarity_search(user_input)
    prompt = PromptTemplate(input_variables=["question"], template="다음은 'ALL-in 코딩 공모전' 수상작 요약입니다. 질문에 대해 답변해주세요: {question}")
    formatted_prompt = prompt.format(question=user_input)
    
    # ChatOpenAI 모델이 필요로 하는 HumanMessage 형식으로 변환
    answer = llm([HumanMessage(content=formatted_prompt)])  # 입력을 ChatMessage로 전달
    st.write("**챗봇의 답변:**", answer.content)
    
    # 대화 기록 저장
    with open("conversation_log.txt", "a") as f:
        f.write(f"사용자: {user_input}\n챗봇: {answer.content}\n\n")
