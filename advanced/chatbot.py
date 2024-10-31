import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Model and sentiment analysis setup
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Memory setup to maintain conversation context
memory = ConversationBufferMemory()

# Streamlit UI setup
st.title("🌸 마음 쉼터 상담 챗봇 🌸")
st.write("안녕하세요! 언제든지 마음의 부담을 나눠보세요. 따뜻하게 귀 기울여 드리겠습니다.")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI styling with CSS
st.markdown("""
    <style>
    .user-message { background-color: #E0F7FA; padding: 10px; border-radius: 10px; margin: 10px 0;}
    .assistant-message { background-color: #FFF3E0; padding: 10px; border-radius: 10px; margin: 10px 0;}
    .container { max-width: 650px; margin: auto;}
    </style>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    style_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(f"<div class='{style_class}'>{content}</div>", unsafe_allow_html=True)

# Process user input
if prompt := st.chat_input("저에게 본인의 마음을 털어놓아보세요..."):
    # Perform sentiment analysis
    sentiment_result = sentiment_analyzer(prompt)[0]
    sentiment = sentiment_result['label']
    
    # Adjust tone based on sentiment
    if sentiment == "1 star" or sentiment == "2 stars":
        tone = "따뜻하고 공감적인"
    elif sentiment == "4 stars" or sentiment == "5 stars":
        tone = "긍정적이고 따뜻한"
    else:
        tone = "편안한"

    # Save user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create the conversational prompt including memory context
    conversation_history = memory.load_memory_variables({}).get("history", "")
    question_template = PromptTemplate(
        input_variables=["tone", "conversation_history", "user_input"],
        template="{tone} 말투로 답변해 주세요. 이전 대화: {conversation_history} 사용자 질문: {user_input}"
    )
    formatted_prompt = question_template.format(
        tone=tone, conversation_history=conversation_history, user_input=prompt
    )

    # Generate response using GPT-4 model
    answer = llm([HumanMessage(content=formatted_prompt)]).content
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save conversation context
    memory.save_context({"input": prompt}, {"output": answer})
