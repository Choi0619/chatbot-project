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
    main_content = soup.find("section", class_="css-18vt64m")  # 필요한 HTML 태그 지정
    
    # main_content가 존재하지 않을 경우 처리
    if main_content:
        return main_content
    else:
        return "Error: The main content could not be found. Please check the HTML structure."

# 수상작 정보 추출 함수
def extract_award_info(soup):
    awards = []
    award_sections = soup.find_all('h2')  # 수상 제목이 들어간 h2 태그를 모두 찾음
    
    for award in award_sections:
        award_title = award.get_text()
        project_name = award.find_next('h2').get_text()
        creators = award.find_next('p').get_text()
        
        # 상세 설명 추출 (가능한 경우)
        description_block = award.find_next('div', class_="my-callout")
        description = description_block.get_text() if description_block else ""
        
        # 기술 스택 정보 추출
        tech_stack = []
        tech_paragraphs = award.find_all_next('p')
        for paragraph in tech_paragraphs:
            if "사용한 기술 스택" in paragraph.get_text():
                tech_stack.append(paragraph.get_text())
        
        # 수상 정보 리스트에 추가
        awards.append({
            "Award Title": award_title,
            "Project Name": project_name,
            "Creators": creators,
            "Description": description,
            "Tech Stack": tech_stack
        })
    
    return awards

# 블로그 URL에서 수상작 정보 추출
url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
soup = fetch_blog_content(url)
award_data = extract_award_info(soup)

# 수상작 정보를 Document로 변환해 저장
documents = [Document(page_content=award["Description"]) for award in award_data]

# 로컬 디렉토리에 저장되는 Chroma 벡터 스토어 설정
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = Chroma.from_documents(documents, embeddings, persist_directory="chroma_store")
vector_store.persist()  # 데이터 저장

# Streamlit 설정 및 챗봇 UI
st.title("All-in Coding Challenge RAG Chatbot")
st.write("이번 챗봇은 'ALL-in 코딩 공모전' 수상작 정보를 요약해주는 RAG 시스템입니다.")

# 기존 대화 내용 저장을 위한 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사용자 입력을 받는 인터페이스
if prompt := st.chat_input("챗봇에게 질문을 입력하세요:"):
    # 사용자 메시지를 화면에 표시하고 기록에 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 사용자 질문에 따라 수상작 정보를 검색하고 응답 생성
    docs = vector_store.similarity_search(prompt)
    question_template = PromptTemplate(input_variables=["question"], template="{question}에 대한 답변입니다.")
    formatted_prompt = question_template.format(question=prompt)
    
    # 모델 응답 생성
    answer = llm([HumanMessage(content=formatted_prompt)])
    
    # 수상작 정보를 바탕으로 응답 구성
    answer_content = ""
    for doc in docs:
        answer_content += doc.page_content + "\n\n"

    # 챗봇 응답을 화면에 표시하고 기록에 저장
    with st.chat_message("assistant"):
        st.markdown(answer_content if answer_content else answer.content)
    st.session_state.messages.append({"role": "assistant", "content": answer_content if answer_content else answer.content})

    # 대화 기록을 파일로 저장
    with open("conversation_log.txt", "a") as f:
        f.write(f"사용자: {prompt}\n챗봇: {answer_content if answer_content else answer.content}\n\n")
