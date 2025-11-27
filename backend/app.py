import os
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
import httpx

# ============================================================
#  FASTAPI SETUP (do this FIRST, before any env checks)
# ============================================================
app = FastAPI(title="Rules Assistant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  DATA PATHS (Railway structure)
# ============================================================
BASE_DIR = Path(__file__).parent.parent  # Go up from backend/ to root
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

DB_PATH = DATA_DIR / "chunks.db"
INDEX_PATH = DATA_DIR / "rulebook.index"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# ============================================================
#  LAZY INITIALIZATION (defer until first request)
# ============================================================
_client = None
_index = None
_conn = None
_system_prompt = None

def get_openai_client():
    """Lazy load OpenAI client - checks env var at request time, not import time"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="OPENAI_API_KEY not configured. Add it in Railway Variables tab."
            )
        http_client = httpx.Client(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40)
        )
        _client = OpenAI(api_key=api_key, http_client=http_client)
    return _client

def get_index():
    """Lazy load FAISS index"""
    global _index
    if _index is None:
        if not INDEX_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Index not found: {INDEX_PATH}")
        _index = faiss.read_index(str(INDEX_PATH))
    return _index

def get_db_conn():
    """Lazy load SQLite connection"""
    global _conn
    if _conn is None:
        if not DB_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Database not found: {DB_PATH}")
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn

def get_system_prompt():
    """Lazy load system prompt"""
    global _system_prompt
    if _system_prompt is None:
        if SYSTEM_PROMPT_PATH.exists():
            _system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        else:
            _system_prompt = """You are an expert rules assistant for a tabletop game.
You must answer ONLY using the provided rulebook excerpts.

Format:
SHORT ANSWER: [Clear answer with brief mechanical context]

REASONING: [Detailed explanation with citations]

If the excerpts don't contain the answer:
SHORT ANSWER: No explicit rule found.
REASONING: The rulebook sections provided do not cover this specific situation."""
    return _system_prompt

# ============================================================
#  SESSION MEMORY
# ============================================================
last_questions = {}

# ============================================================
#  DATA MODELS
# ============================================================
class Source(BaseModel):
    page: int
    image_url: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: List[Source]

# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def embed(text: str) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def rewrite_query(user_message: str, session_id: Optional[str]) -> str:
    """Rewrite casual questions into rulebook search terms."""
    client = get_openai_client()
    text = user_message.strip()
    is_short = (
        len(text.split()) <= 6
        or text.lower().startswith(("what about", "what if", "and if", "and what", "how about"))
    )
    
    if session_id and is_short and session_id in last_questions:
        combined = f"{last_questions[session_id]}\nFollow-up: {text}"
    else:
        combined = text
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Rewrite game rules questions into concise search queries. Keep key terms. Output ONLY the query."
            },
            {"role": "user", "content": combined}
        ],
    )
    return completion.choices[0].message.content.strip()

def fetch_chunks(indices: List[int]) -> List[Dict]:
    """Fetch chunks from SQLite by FAISS indices."""
    conn = get_db_conn()
    chunk_data = []
    for faiss_idx in indices:
        sqlite_id = faiss_idx + 1  # FAISS 0-indexed, SQLite 1-indexed
        row = conn.execute("SELECT page, content FROM chunks WHERE id=?", (sqlite_id,)).fetchone()
        if row:
            chunk_data.append({"page": row["page"], "content": row["content"]})
    return chunk_data

def generate_answer(question: str, rewritten: str, chunks: List[Dict]) -> str:
    client = get_openai_client()
    combined_text = "\n\n".join(f"[p{c['page']}] {c['content']}" for c in chunks)
    
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {
            "role": "user",
            "content": f"QUESTION: {question}\nSEARCH TERMS: {rewritten}\n\nRULEBOOK EXCERPTS:\n{combined_text}"
        }
    ]
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# ============================================================
#  ROUTES
# ============================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    
    # Get index (lazy loaded)
    index = get_index()
    
    # Rewrite query
    rewritten = rewrite_query(req.message, session_id)
    
    # Embed and search
    emb1 = embed(req.message)
    emb2 = embed(rewritten)
    avg_emb = ((emb1 + emb2) / 2).astype("float32")
    
    D, I = index.search(np.array([avg_emb]), 8)
    indices = I[0].tolist()
    
    # Get chunks
    chunks = fetch_chunks(indices)
    
    if not chunks:
        return ChatResponse(
            session_id=session_id,
            reply="SHORT ANSWER: No rule found.\n\nREASONING: I couldn't find anything related to that in the rulebook.",
            sources=[]
        )
    
    # Generate answer
    answer = generate_answer(req.message, rewritten, chunks)
    
    # Save for follow-up context
    last_questions[session_id] = req.message
    
    # Prepare sources
    unique_pages = sorted(set(c["page"] for c in chunks))
    sources = [Source(page=p) for p in unique_pages]
    
    return ChatResponse(session_id=session_id, reply=answer, sources=sources)

@app.get("/health")
def health():
    """Health check - also verifies configuration"""
    status = {"status": "healthy"}
    
    # Check if API key is set (without revealing it)
    if os.getenv("OPENAI_API_KEY"):
        status["openai"] = "configured"
    else:
        status["openai"] = "NOT CONFIGURED - add OPENAI_API_KEY in Railway Variables"
        status["status"] = "degraded"
    
    # Check data files
    status["index_exists"] = INDEX_PATH.exists()
    status["db_exists"] = DB_PATH.exists()
    status["frontend_exists"] = FRONTEND_DIR.exists()
    
    # Show paths for debugging
    status["paths"] = {
        "base": str(BASE_DIR),
        "data": str(DATA_DIR),
        "frontend": str(FRONTEND_DIR)
    }
    
    return status

@app.get("/debug/env")
def debug_env():
    """Debug endpoint to check environment (doesn't reveal secrets)"""
    return {
        "OPENAI_API_KEY_SET": bool(os.getenv("OPENAI_API_KEY")),
        "RAILWAY_ENVIRONMENT": os.getenv("RAILWAY_ENVIRONMENT_NAME", "not set"),
        "RAILWAY_SERVICE": os.getenv("RAILWAY_SERVICE_NAME", "not set"),
        "PWD": os.getcwd(),
        "files_in_app": os.listdir("/app") if os.path.exists("/app") else "no /app dir"
    }

# ============================================================
#  SERVE FRONTEND (must be LAST - catches all other routes)
# ============================================================
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def no_frontend():
        return {"error": f"Frontend not found at {FRONTEND_DIR}", "hint": "Check your folder structure"}

print("Rules Assistant API ready!")
print(f"  BASE_DIR: {BASE_DIR}")
print(f"  DATA_DIR: {DATA_DIR}")
print(f"  FRONTEND_DIR: {FRONTEND_DIR}")
print(f"  OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
