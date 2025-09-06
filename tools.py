# backend/tools.py
import os
import requests
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ---- CONFIG ----
# Path to your saved FAISS index directory (change if needed)
FAISS_PATH = os.getenv("FAISS_PATH", "backend/vectorstore/faiss_index")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ---- Retriever loader ----
def load_retriever(k: int = 4):
    """
    Load local FAISS vectorstore and return a retriever object.
    WARNING: allow_dangerous_deserialization=True is required when loading pickled metadata.
    Only enable if you trust the vectorstore file.
    """
    db = FAISS.load_local(
        FAISS_PATH,
        embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": k})

def retrieve_documents_text(query: str, k: int = 4) -> str:
    """
    Return concatenated text snippets from top-k retrieved docs.
    We return short excerpts so prompts remain compact.
    """
    retriever = load_retriever(k=k)
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."

    pieces = []
    for i, d in enumerate(docs, start=1):
        content = getattr(d, "page_content", str(d))
        src = d.metadata.get("source") if hasattr(d, "metadata") else None
        header = f"[doc {i} {src}]" if src else f"[doc {i}]"
        # Trim content to keep prompt size manageable
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 800:
            snippet = snippet[:800].rsplit(" ", 1)[0] + "..."
        pieces.append(f"{header} {snippet}")
    return "\n\n".join(pieces)

# ---- Weather tool (placeholder) ----
def fetch_weather(location: str) -> str:
    """
    Fetch weather for a location. Replace with a real API key to enable.
    If OPENWEATHER_API_KEY is not set, returns a helpful placeholder.
    """
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return f"[Weather tool not configured] Requested weather for: {location}"
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": key, "units": "metric"},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"{location}: {desc}, {temp}°C"
    except Exception as e:
        return f"Weather API error: {e}"

# ---- Market price tool (placeholder) ----
def fetch_market_price(query: str) -> str:
    """
    Placeholder for fetching market prices.
    Input examples:
      - "tomato | Bangalore"
      - "tomato"
    Replace with a real Agmarknet / local API integration.
    """
    return f"[Market tool not configured] Requested market info for: {query}"

# ---- Simple calculator (safe) ----
def safe_calc(expression: str) -> str:
    """
    Very limited safe arithmetic evaluator.
    Accepts digits, + - * / ( ) . and spaces only.
    """
    import re
    expr = expression.strip()
    if not expr:
        return "No expression provided."
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return "ERROR: Only basic arithmetic expressions allowed (digits and +-*/())."
    try:
        # Evaluate in a restricted environment
        result = eval(expr, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
