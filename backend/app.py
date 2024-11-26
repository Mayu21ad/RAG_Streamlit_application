import streamlit as st
import os
from groq import Groq
import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


def main():
        
    env_path = os.path.join('.env')
    load_dotenv(dotenv_path=env_path)
    groq_api_key = os.getenv('GROQ_API_KEY')

    st.title("CHAT RAG APPLICATION !!!")
    st.write("A user friendly groq chatbot")
    st.sidebar.title("Customizable with LLMs")

    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama3-8b-8192']
    )

    conversational_memory_length = st.sidebar.slider('Conversational memory length :', 1, 10, value=5)

    memory = ConversationBufferMemory(k=conversational_memory_length)

    user_question = st.text_input("ASK:")

    if 'chat_history' not in st.session_state: # SESSION STATE
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input' :message['human']},{'output' :message['AI']})

    groq_chat = ChatGroq(  # OBJECT FOR LANGCHAIN AND CONVERSATION
        groq_api_key=groq_api_key,
        model_name=model
    )

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )

    if user_question:
        response=conversation(user_question)
        message={'human':user_question, 'AI':response['response']}
        st.session_state.chat_history.append(message)
        st.write("ChatBot:", response['response'])

if __name__ == "__main__":
    main()
