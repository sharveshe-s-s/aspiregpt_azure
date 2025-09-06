import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ⚠️ Use SAME embeddings as used during FAISS build:
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

db = FAISS.load_local(
    "faiss_index",  # folder path to vectorstore
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are AgriGPT, an AI agricultural expert that always responds in the *same language as the question*. your
role is to help the farmers of india. Use the given context to answer accurately. if the context is not
enough you can always use your knowledge base. Maintain the original language throughout the answer.

Context:
{context}

Question:
{question}


Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2),
    retriever=retriever,
    return_source_documents=False,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

def get_qa_chain():
    return qa_chain
