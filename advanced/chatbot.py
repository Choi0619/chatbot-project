import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from transformers import pipeline

# 환경 변수 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 모델 및 감정 분석 파이프라인 설정
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Memory 설정으로 대화 컨텍스트 유지
memory = ConversationBufferMemory()

# Streamlit UI 설정
st.title("🌸 마음 쉼터 상담 챗봇 🌸")
st.write("안녕하세요! 언제든지 마음의 부담을 나눠보세요. 따뜻하게 귀 기울여 드리겠습니다.")

# 기존 대화 내용 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI 스타일: CSS
st.markdown("""
    <style>
    .user-message { background-color: #E0F7FA; padding: 10px; border-radius: 10px; margin: 10px 0;}
    .assistant-message { background-color: #FFF3E0; padding: 10px; border-radius: 10px; margin: 10px 0;}
    .container { max-width: 650px; margin: auto;}
    </style>
""", unsafe_allow_html=True)

# 대화 내용 출력
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    style_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(f"<div class='{style_class}'>{content}</div>", unsafe_allow_html=True)

# 사용자 입력 처리
if prompt := st.chat_input("저에게 본인의 마음을 털어놓아보세요..."):
    # 감정 분석 수행
    sentiment_result = sentiment_analyzer(prompt)[0]
    sentiment = sentiment_result['label']
    
    # 감정에 따른 프롬프트 조정
    if sentiment == "1 star" or sentiment == "2 stars":
        tone = "차분하고 다정하게"
    elif sentiment == "4 stars" or sentiment == "5 stars":
        tone = "긍정적이고 따뜻하게"
    else:
        tone = "중립적이고 편안하게"
    
    # 사용자 입력 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 이전 대화를 포함한 프롬프트 생성
    conversation_history = memory.load_memory_variables({}).get("history", "")
    formatted_prompt = f"{tone} 답변해 주세요: {prompt}\n\n{conversation_history}"

    # GPT-4 응답 생성
    answer = llm([HumanMessage(content=formatted_prompt)]).content
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 대화 저장
    memory.save_context({"input": prompt}, {"output": answer})
