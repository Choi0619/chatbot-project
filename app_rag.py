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
        return main_content
    else:
        return "Error: The main content could not be found. Please check the HTML structure."

# 수상작 정보 추출 함수
def extract_award_info(soup):
    awards = []
    award_sections = soup.find_all('h2')  # 수상 제목이 들어간 h2 태그를 모두 찾음
    
    for award in award_sections:
        award_title = award.get_text()
        project_name_element = award.find_next('h2')
        project_name = project_name_element.get_text() if project_name_element else "프로젝트 이름을 찾을 수 없음"
        
        creators_element = award.find_next('p')
        creators = creators_element.get_text() if creators_element else "참여자 정보를 찾을 수 없음"
        
        # 기술 스택 정보 추출 (요약에 포함)
        tech_stack = []
        tech_paragraphs = award.find_all_next('p')
        for paragraph in tech_paragraphs:
            if "사용한 기술 스택" in paragraph.get_text():
                tech_stack.append(paragraph.get_text())
                break  # 첫 번째 기술 스택 정보만 포함
        
        # 수상 정보 리스트에 추가
        awards.append({
            "Award Title": award_title,
            "Project Name": project_name,
            "Creators": creators,
            "Tech Stack": tech_stack[0] if tech_stack else "기술 스택 정보를 찾을 수 없음"  # 요약된 기술 스택 정보
        })
    
    return awards

# 블로그 URL에서 수상작 정보 추출
url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
soup = fetch_blog_content(url)
award_data = extract_award_info(soup)

# Streamlit 설정 및 챗봇 UI
st.title("All-in Coding Challenge RAG Chatbot")
st.write("이번 챗봇은 'ALL-in 코딩 공모전' 수상작 정보를 요약해주는 RAG 시스템입니다.")

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

    # 사용자 질문에 따라 수상작 정보를 검색하고 요약 생성
    if "ALL-in 코딩 공모전 수상작들을 요약해줘" in prompt:
        # 요약된 수상작 정보 생성
        answer_content = "All-in 코딩 공모전 수상작 요약:\n\n"
        for award in award_data:
            answer_content += f"🏆 {award['Award Title']} - {award['Project Name']}\n"
            answer_content += f"제작자: {award['Creators']}\n"
            answer_content += f"기술 스택: {award['Tech Stack']}\n\n"
        
        with st.chat_message("assistant"):
            st.markdown(answer_content)
        st.session_state.messages.append({"role": "assistant", "content": answer_content})

    else:
        # RAG 시스템을 사용해 응답 생성
        docs = vector_store.similarity_search(prompt)
        question_template = PromptTemplate(input_variables=["question"], template="{question}에 대한 답변입니다.")
        formatted_prompt = question_template.format(question=prompt)
        
        # 모델 응답 생성
        answer = llm([HumanMessage(content=formatted_prompt)])
        with st.chat_message("assistant"):
            st.markdown(answer.content)
        st.session_state.messages.append({"role": "assistant", "content": answer.content})

    # 대화 기록을 파일로 저장
    with open("conversation_log.txt", "a") as f:
        f.write(f"사용자: {prompt}\n챗봇: {answer_content if 'ALL-in 코딩 공모전 수상작들을 요약해줘' in prompt else answer.content}\n\n")
