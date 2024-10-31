import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from transformers import pipeline

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 모델과 감정 분석 설정
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 대화 맥락 유지를 위한 메모리 설정
memory = ConversationBufferMemory()

# Streamlit UI 설정
st.title("🌸 마음 쉼터 상담 챗봇 🌸")
st.write("안녕하세요! 언제든지 마음의 부담을 나눠보세요. 따뜻하게 귀 기울여 드리겠습니다.")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI 스타일링 CSS 적용
st.markdown("""
    <style>
    .user-message { background-color: #E0F7FA; padding: 10px; border-radius: 10px; margin: 10px 0;}
    .assistant-message { background-color: #FFF3E0; padding: 10px; border-radius: 10px; margin: 10px 0;}
    .container { max-width: 650px; margin: auto;}
    </style>
""", unsafe_allow_html=True)

# 채팅 기록 표시
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    style_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(f"<div class='{style_class}'>{content}</div>", unsafe_allow_html=True)

# 사용자 입력 처리
if prompt := st.chat_input("저에게 본인의 마음을 털어놓아보세요..."):
    # 감정 분석 수행
    sentiment_result = sentiment_analyzer(prompt)[0]
    sentiment = sentiment_result['label']

    # 감정 분석 결과에 따라 톤 설정
    if sentiment in ["1 star", "2 stars"]:  # 부정적인 감정
        tone = "위로와 공감을 드리는 존댓말로"
    elif sentiment in ["4 stars", "5 stars"]:  # 긍정적인 감정
        tone = "따뜻하고 격려하는 존댓말로"
    else:  # 중립적인 감정
        tone = "편안하고 공감적인 존댓말로"

    # 사용자 입력 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 대화 맥락을 포함하여 대화 프롬프트 생성
    conversation_history = memory.load_memory_variables({}).get("history", "")
    question_template = PromptTemplate(
        input_variables=["tone", "conversation_history", "user_input"],
        template="{tone} 답변해 주세요. 이전 대화: {conversation_history} 사용자 질문: {user_input}"
    )
    formatted_prompt = question_template.format(
        tone=tone, conversation_history=conversation_history, user_input=prompt
    )

    # GPT-4 모델을 사용하여 응답 생성
    answer = llm([HumanMessage(content=formatted_prompt)]).content
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 대화 맥락 저장
    memory.save_context({"input": prompt}, {"output": answer})
