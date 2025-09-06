# backend/main.py
import os
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import openai
import requests

# Embeddings & FAISS from langchain-community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient

# ---------- Load .env from explicit path ----------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    logging.warning(f".env file not found at {ENV_PATH}, relying on environment variables.")

# Optional: whisper for server-side STT (if installed)
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ---------- CONFIG (from environment variables) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # REQUIRED
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")  # REQUIRED for FAISS
FAISS_CONTAINER_NAME = os.getenv("FAISS_CONTAINER_NAME", "faiss")
FAISS_INDEX_LOCAL = os.getenv("FAISS_INDEX_PATH", "faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
PORT = int(os.getenv("PORT", 10000))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")  # set to your frontend origin in production

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agri-backend")

# Validate required config
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set. Exiting.")
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
if not AZURE_CONNECTION_STRING:
    logger.error("AZURE_CONNECTION_STRING environment variable is not set. Exiting.")
    raise RuntimeError("AZURE_CONNECTION_STRING is required to load FAISS from Blob Storage.")

openai.api_key = OPENAI_API_KEY

app = FastAPI(title="PathFinder AI")

# CORS for frontend access
allow_origins_list = [ALLOW_ORIGINS] if ALLOW_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (audio)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------- Load embeddings and FAISS vectorstore ----------
logger.info("Initializing embeddings with model: %s", EMBEDDING_MODEL)
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
except Exception as e:
    logger.exception("Failed to initialize embeddings: %s", e)
    embeddings = None

vectorstore = None

def download_faiss_from_blob(local_path: str):
    """Download FAISS index files from Azure Blob Storage if not present locally."""
    if os.path.exists(local_path):
        logger.info("FAISS index already exists locally at %s", local_path)
        return
    os.makedirs(local_path, exist_ok=True)
    logger.info("Downloading FAISS index from Azure Blob Storage to %s", local_path)
    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(FAISS_CONTAINER_NAME)
    for blob_name in ["index.faiss", "index.pkl"]:
        blob_client = container_client.get_blob_client(blob_name)
        out_file = os.path.join(local_path, blob_name)
        with open(out_file, "wb") as f:
            f.write(blob_client.download_blob().readall())
        logger.info("Downloaded %s", blob_name)

if embeddings is not None:
    try:
        download_faiss_from_blob(FAISS_INDEX_LOCAL)
        vectorstore = FAISS.load_local(FAISS_INDEX_LOCAL, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load FAISS index. Reason: %s", e)
        vectorstore = None
else:
    logger.warning("Embeddings not initialized; skipping FAISS load.")

# Load whisper model optionally
whisper_model = None
if WHISPER_AVAILABLE:
    try:
        logger.info("Loading Whisper model (may take time)...")
        whisper_model = whisper.load_model("base")
    except Exception as e:
        logger.exception("Failed to load Whisper model: %s", e)
        whisper_model = None
else:
    logger.info("Whisper not available. /voice will return an explanatory error.")

# --------- Request model ----------
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "en"  # language code from frontend

# --------- Utilities ----------
def call_openai_chat(system_prompt: str, user_prompt: str) -> str:
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        raise

def generate_tts(text: str, lang: str = "en", filename_prefix: str = "answer") -> str:
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed on server.")
    fname = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.mp3"
    out_path = os.path.join("static", fname)
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(out_path)
    return f"/static/{fname}"

# --------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "PathFinder AI backend running"}

@app.post("/get_answer/")
async def get_answer(req: QueryRequest):
    if vectorstore is None:
        logger.error("FAISS vectorstore not loaded.")
        raise HTTPException(status_code=500, detail="FAISS vectorstore not loaded on server.")

    user_query = (req.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(user_query)
        context_text = "\n\n".join([d.page_content for d in docs]) if docs else ""
    except Exception as e:
        logger.exception("Retrieval error: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    realtime_info = ""
    if any(w in user_query.lower() for w in ["weather", "temperature", "rain", "forecast"]):
        # Replace this with your actual weather function if needed
        realtime_info = ""  

    system_prompt = (
        """You are CareerGPT, a personalized AI career and skills advisor for Indian students use the supplied context to answer accurately. 
         - Understand the student’s background, skills, and interests.
         - Recommend suitable career paths with clear reasons.
         - List specific, actionable skills required to succeed.
         - Suggest learning resources (courses, certifications, internships).
         - Adapt to the evolving job market, focusing on future-proof roles.
           Always respond in the same language as the user’s question."""
    )

    user_prompt_parts = [f"Question: {user_query}"]
    if context_text:
        user_prompt_parts.append(f"Context:\n{context_text}")
    if realtime_info:
        user_prompt_parts.append(f"RealTimeData:\n{realtime_info}")

    user_prompt_parts.append("Answer using the context if relevant. If context doesn't contain the answer, use general career related knowledge.")
    user_prompt = "\n\n".join(user_prompt_parts)

    try:
        answer_text = call_openai_chat(system_prompt, user_prompt)
    except Exception as e:
        logger.exception("OpenAI error: %s", e)
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    audio_url = None
    if GTTS_AVAILABLE:
        lang_code = (req.language or "en")
        fallback = {"mni": "en"}
        tts_lang = fallback.get(lang_code, lang_code)
        try:
            audio_url = generate_tts(answer_text, lang=tts_lang, filename_prefix="answer")
        except Exception as e:
            logger.warning("TTS generation failed: %s", e)
            audio_url = None

    return {"answer": answer_text, "audio_url": audio_url}

@app.post("/voice/")
async def voice_transcribe(file: UploadFile = File(...)):
    if not WHISPER_AVAILABLE or whisper_model is None:
        logger.info("Whisper not installed or model not loaded.")
        return {"error": "Server-side transcription not available. Whisper not installed."}
    try:
        content = await file.read()
        tmp = f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        with open(tmp, "wb") as f:
            f.write(content)
        result = whisper_model.transcribe(tmp)
        text = result.get("text", "").strip()
        try:
            os.remove(tmp)
        except Exception:
            pass
        return {"text": text}
    except Exception as e:
        logger.exception("Voice transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app" if __package__ else "main:app", host="0.0.0.0", port=PORT)
