# import streamlit as st
# import os
# from langchain.chains import LLMChain
# from langchain_core.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
# )
# from langchain_core.messages import SystemMessage
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain_groq import ChatGroq

# def main():
#     """
#     Main function to run the Streamlit app for chatting with Groq.
#     """
#     # Load the Groq API key
#     groq_api_key = os.getenv('GROQ_API_KEY')
#     if not groq_api_key:
#         st.error("Error: GROQ_API_KEY is not set. Please ensure it is defined in your environment variables.")
#         return

#     # Display header and sidebar
#     st.title("Chat with Groq!")
#     st.write("Hello! I'm your friendly Groq chatbot. Let's start our conversation!")

#     # Sidebar customization options
#     st.sidebar.title('Customization')
#     system_prompt = st.sidebar.text_input("System prompt:", value="You are a helpful assistant.")
#     model = st.sidebar.selectbox(
#         'Choose a model',
#         ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
#     )
#     conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

#     # Initialize conversational memory
#     memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

#     # Load chat history from session state
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     else:
#         for message in st.session_state.chat_history:
#             memory.save_context(
#                 {'input': message['human']},
#                 {'output': message['AI']}
#             )

#     # Text input for user question
#     user_question = st.text_input("Ask a question:")

#     # Initialize the Groq LangChain chat object
#     groq_chat = ChatGroq(
#         groq_api_key=groq_api_key, 
#         model_name=model
#     )

#     # Respond to user input
#     if user_question:
#         # Create a prompt template
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 SystemMessage(content=system_prompt),
#                 MessagesPlaceholder(variable_name="chat_history"),
#                 HumanMessagePromptTemplate.from_template("{human_input}"),
#             ]
#         )

#         # Create a conversation chain
#         conversation = LLMChain(
#             llm=groq_chat,
#             prompt=prompt,
#             verbose=True,
#             memory=memory,
#         )

#         # Generate response
#         response = conversation.predict(human_input=user_question)
#         message = {'human': user_question, 'AI': response}

#         # Update session state
#         st.session_state.chat_history.append(message)

#         # Display chatbot response
#         st.markdown(f"**Chatbot:** {response}")

# if __name__ == "__main__":
#     main()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
import os
# from groq import Groq
# import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

env_path = os.path.join('.env')
load_dotenv(dotenv_path=env_path)
groq_api_key = os.getenv('GROQ_API_KEY')

app = FastAPI()

memory = ConversationBufferMemory(k=5)

class ChatRequest(BaseModel):
    model: str
    memory_length: int
    question: str
    chat_history: list[dict]

class ChatResponse(BaseModel):
    response: str

# Helper function to create Groq Chat
def create_groq_chat(model):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    # Initialize memory with the specified length
    memory = ConversationBufferMemory(k=request.memory_length)

    # Save previous chat history in memory
    for message in request.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Create Groq Chat object
    groq_chat = create_groq_chat(request.model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Get the response
    response = conversation.run(request.question)
    return ChatResponse(response=response)


# def main():
        
   

#     st.title("CHAT RAG APPLICATION !!!")
#     st.write("A user friendly groq chatbot")
#     st.sidebar.title("Customizable with LLMs")

#     model = st.sidebar.selectbox(
#         'Choose a model',
#         ['mixtral-8x7b-32768', 'llama3-8b-8192']
#     )

#     conversational_memory_length = st.sidebar.slider('Conversational memory length :', 1, 10, value=5)

#     memory = ConversationBufferMemory(k=conversational_memory_length)

#     user_question = st.text_input("ASK:")

#     if 'chat_history' not in st.session_state: # SESSION STATE
#         st.session_state.chat_history=[]
#     else:
#         for message in st.session_state.chat_history:
#             memory.save_context({'input' :message['human']},{'output' :message['AI']})

#     groq_chat = ChatGroq(  # OBJECT FOR LANGCHAIN AND CONVERSATION
#         groq_api_key=groq_api_key,
#         model_name=model
#     )

#     conversation = ConversationChain(
#         llm=groq_chat,
#         memory=memory
#     )

#     if user_question:
#         response=conversation(user_question)
#         message={'human':user_question, 'AI':response['response']}
#         st.session_state.chat_history.append(message)
#         st.write("ChatBot:", response['response'])

# if __name__ == "__main__":
#     main()
