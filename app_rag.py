import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
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
    main_content = soup.find("section", class_="css-18vt64m")
    
    # main_content가 존재하지 않을 경우 처리
    if main_content:
        paragraphs = main_content.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        return content
    else:
        return "Error: The main content could not be found. Please check the HTML structure."

# 블로그 URL에서 내용 추출 및 분할
url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
content = fetch_blog_content(url)
documents = [Document(page_content=content)]

# 임베딩 저장
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = Chroma.from_documents(documents, embeddings, persist_directory="chroma_store")
vector_store.persist()

# Streamlit 설정 및 챗봇 UI
st.title("All-in Coding Challenge RAG Chatbot")

# 기존 대화 내용 저장을 위한 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 내용을 표시하기
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# 사용자 입력을 받는 인터페이스
if prompt := st.chat_input("챗봇에게 질문을 입력하세요:"):
    # 사용자 메시지를 화면에 표시하고 기록에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG 시스템을 사용해 응답 생성
    docs = vector_store.similarity_search(prompt, k=3)  # 가장 관련된 상위 3개의 문서만 요약 생성에 사용
    summarized_content = "\n".join([doc.page_content for doc in docs])
    
    question_template = PromptTemplate(
        input_variables=["context", "question"],
        template="다음 내용의 핵심을 요약해 답변해줘: {context} 질문: {question}"
    )
    formatted_prompt = question_template.format(context=summarized_content, question=prompt)
    
    # 모델 응답 생성
    answer = llm([HumanMessage(content=formatted_prompt)])
    with st.chat_message("assistant"):
        st.markdown(answer.content)
    st.session_state.messages.append({"role": "assistant", "content": answer.content})

    # 대화 기록을 파일로 저장
    with open("conversation_log.txt", "a") as f:
        f.write(f"사용자: {prompt}\n챗봇: {answer.content}\n\n")
