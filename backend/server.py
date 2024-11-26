from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join('.env')
load_dotenv(dotenv_path=env_path)

# Ensure the GROQ API key is present
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")

# Initialize FastAPI app
app = FastAPI()

# Initialize resources
loader = WebBaseLoader("https://www.kotaksecurities.com/investing-guide/share-market/how-to-trade-in-stock-for-beginners/#close-modal")
docs = loader.load()
embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
vector_store = FAISS.from_documents(documents, embeddings)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768'
)

prompt_template = ChatPromptTemplate("""
    Answer the following questions based on the provided context in detail.
    <context>
    {context}
    </context>

    Question: {input}"""
)
doc_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

# Pydantic model for input validation
class QueryRequest(BaseModel):
    prompt: str

# Define routes
@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        response = retrieval_chain.invoke({"input": request.prompt})
        return {
            "answer": response.get("answer", "No answer found."),
            "context": [doc.page_content for doc in response.get("context", [])],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
