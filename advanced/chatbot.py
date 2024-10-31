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

# Streamlit UI 설정 - 페이지 타이틀
st.set_page_config(page_title="마음 쉼터 상담 챗봇", page_icon="🌸")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI 시작 - 챗봇 타이틀 및 설명
st.title("🌸 마음 쉼터 상담 챗봇 🌸")
st.write("안녕하세요! 따뜻한 마음으로 귀 기울여 드릴게요. 언제든지 마음을 나눠보세요.")

# 채팅 기록 표시 - 기본 Streamlit 스타일 사용
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            st.write(content)

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
        st.write(prompt)

    # 대화 맥락을 포함하여 대화 프롬프트 생성
    conversation_history = memory.load_memory_variables({}).get("history", "")
    # 대화체 스타일로 답변을 유도하는 프롬프트 템플릿
    question_template = PromptTemplate(
        input_variables=["tone", "conversation_history", "user_input"],
        template="{tone} 말투로, 넘버링 없이 마치 친구가 이야기하듯 편하게 조언해 주세요. 예를 들어, '저도 가벼운 산책이나 운동을 할 때 기분이 많이 나아지더라고요.' 같은 방식으로 답변을 작성해 주세요. 이전 대화: {conversation_history} 사용자 질문: {user_input}"
    )

    formatted_prompt = question_template.format(
        tone=tone, conversation_history=conversation_history, user_input=prompt
    )

    # GPT-4 모델을 사용하여 응답 생성
    answer = llm([HumanMessage(content=formatted_prompt)]).content
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 대화 맥락 저장
    memory.save_context({"input": prompt}, {"output": answer})

# 상담 종료 버튼 및 피드백 창
if st.button("상담 종료"):
    st.subheader("상담이 도움이 되셨나요?")
    feedback = st.radio("상담 경험을 평가해주세요:", ("매우 만족", "만족", "보통", "불만족", "매우 불만족"))
    if feedback:
        st.success("피드백을 주셔서 감사합니다! 상담 챗봇의 개선에 도움이 됩니다.")
