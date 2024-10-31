import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from transformers import pipeline

# 환경 변수 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Streamlit UI 설정
st.title("마음 쉼터 상담 챗봇")
st.write("심리적 안정감을 주는 상담 챗봇입니다. 언제든지 마음의 부담을 나눠보세요.")

# 기존 대화 내용 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 내용 출력
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    with st.chat_message(role):
        st.markdown(content)

# 사용자 입력 처리
if prompt := st.chat_input("저에게 본인의 마음을 털어놓아보세요..."):
    # 감정 분석 수행
    sentiment_result = sentiment_analyzer(prompt)[0]
    sentiment = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    # 감정에 따른 프롬프트 조정
    if sentiment == "NEGATIVE":
        tone = "따뜻한 느낌을 주는 다정한 상담사의 관점으로"
    elif sentiment == "POSITIVE":
        tone = "긍정적이며 따뜻한 관점으로"
    else:
        tone = "중립적이고 편안한 관점으로"

    # 사용자 입력 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # GPT-4 모델에 전달할 프롬프트 생성
    question_template = PromptTemplate(
        input_variables=["tone", "question"],
        template="{tone} 대답해 주세요: {question}"
    )
    formatted_prompt = question_template.format(tone=tone, question=prompt)

    # GPT-4 응답 생성
    answer = llm([HumanMessage(content=formatted_prompt)]).content
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

