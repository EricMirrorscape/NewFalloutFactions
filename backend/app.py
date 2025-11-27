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
from openai import OpenAI
from pydantic import BaseModel
import httpx

# ============================================================
#  FASTAPI SETUP (initialize FIRST, before any env checks)
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
#  PATH CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent.parent  # /app
DATA_DIR = BASE_DIR / "data"              # /app/data
FRONTEND_DIR = BASE_DIR / "frontend"      # /app/frontend
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# ============================================================
#  LAZY-LOADED RESOURCES
# ============================================================
_openai_client = None
_faiss_index = None
_db_connection = None
_system_prompt = None


def get_openai_client():
    """Get OpenAI client, initializing on first call."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not set. Add it in Railway Variables tab."
            )
        _openai_client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(timeout=60.0)
        )
    return _openai_client


def get_faiss_index():
    """Get FAISS index, loading on first call."""
    global _faiss_index
    if _faiss_index is None:
        index_path = DATA_DIR / "rulebook.index"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail=f"Index not found: {index_path}")
        _faiss_index = faiss.read_index(str(index_path))
    return _faiss_index


def get_db_connection():
    """Get SQLite connection, connecting on first call."""
    global _db_connection
    if _db_connection is None:
        db_path = DATA_DIR / "chunks.db"
        if not db_path.exists():
            raise HTTPException(status_code=500, detail=f"Database not found: {db_path}")
        _db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
        _db_connection.row_factory = sqlite3.Row
    return _db_connection


def get_system_prompt():
    """Get system prompt, loading on first call."""
    global _system_prompt
    if _system_prompt is None:
        if SYSTEM_PROMPT_PATH.exists():
            _system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        else:
            _system_prompt = """You are an expert rules assistant for a tabletop game.
Answer ONLY using the provided rulebook excerpts.

Format your response as:
SHORT ANSWER: [Clear, direct answer]

REASONING: [Detailed explanation with page citations]"""
    return _system_prompt


# ============================================================
#  SESSION MEMORY
# ============================================================
last_questions: Dict[str, str] = {}


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
    """Generate embedding for text."""
    client = get_openai_client()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype="float32")


def rewrite_query(question: str, session_id: Optional[str]) -> str:
    """
    Rewrite user question into rulebook search terms.
    
    THIS IS THE CRITICAL FUNCTION - uses detailed prompt with examples
    to transform casual questions into effective search queries.
    """
    client = get_openai_client()
    
    # Build context from previous questions if this is a follow-up
    memory = []
    if session_id and session_id in last_questions:
        memory = [last_questions[session_id]]
    
    if memory:
        history_text = " | ".join(memory[-3:])
        combined = f"Previous questions: {history_text}\nCurrent question: {question}"
    else:
        combined = question
    
    # THE KEY: Detailed prompt with examples - this matches the local version!
    prompt = (
        "Rewrite the tabletop game question into precise rulebook search terms.\n"
        "If it's a follow-up question, incorporate context from previous questions.\n"
        "Focus on game mechanics keywords.\n"
        "Examples:\n"
        "- 'can I shoot his toe' → 'visibility line of sight body parts targeting'\n"
        "- 'what about his hat' (after toe question) → 'visibility clothing ignored targeting'\n"
        "- 'what about his shoulder' (after hat question) → 'visibility body parts shoulder targeting'\n"
        "Output ONLY the rewritten search query."
    )
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": combined}
        ],
        max_tokens=50,
        temperature=0
    )
    
    return resp.choices[0].message.content.strip()


def fetch_chunks(indices: List[int]) -> List[Dict]:
    """Fetch rule chunks from database by FAISS indices."""
    conn = get_db_connection()
    chunks = []
    for faiss_idx in indices:
        sqlite_id = faiss_idx + 1  # FAISS is 0-indexed, SQLite is 1-indexed
        row = conn.execute(
            "SELECT page, content FROM chunks WHERE id=?", 
            (sqlite_id,)
        ).fetchone()
        if row:
            chunks.append({"page": row["page"], "content": row["content"]})
    return chunks


def generate_answer(question: str, rewritten: str, chunks: List[Dict]) -> str:
    """Generate answer using retrieved chunks."""
    client = get_openai_client()
    combined_text = "\n\n".join(f"[p{c['page']}] {c['content']}" for c in chunks)
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n"
                    f"SEARCH TERMS: {rewritten}\n\n"
                    f"RULEBOOK EXCERPTS:\n{combined_text}"
                )
            }
        ]
    )
    return resp.choices[0].message.content.strip()


# ============================================================
#  API ROUTES
# ============================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint."""
    session_id = req.session_id or str(uuid.uuid4())
    
    try:
        # Get FAISS index
        index = get_faiss_index()
        
        # Rewrite query for better search (uses the detailed prompt!)
        rewritten = rewrite_query(req.message, session_id)
        
        # Generate embeddings and search
        emb1 = embed(req.message)
        emb2 = embed(rewritten)
        avg_emb = ((emb1 + emb2) / 2).astype("float32")
        
        D, I = index.search(np.array([avg_emb]), 8)
        indices = I[0].tolist()
        
        # Fetch matching chunks
        chunks = fetch_chunks(indices)
        
        if not chunks:
            return ChatResponse(
                session_id=session_id,
                reply="SHORT ANSWER: No rule found.\n\nREASONING: I couldn't find relevant rules for that question.",
                sources=[]
            )
        
        # Generate answer
        answer = generate_answer(req.message, rewritten, chunks)
        
        # Save for follow-up questions
        last_questions[session_id] = req.message
        
        # Get unique page sources
        unique_pages = sorted(set(c["page"] for c in chunks))
        sources = [Source(page=p) for p in unique_pages]
        
        return ChatResponse(session_id=session_id, reply=answer, sources=sources)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint with diagnostic info."""
    status = {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "data_dir": str(DATA_DIR),
        "index_exists": (DATA_DIR / "rulebook.index").exists(),
        "db_exists": (DATA_DIR / "chunks.db").exists(),
        "frontend_exists": FRONTEND_DIR.exists(),
    }
    
    if not status["openai_configured"]:
        status["status"] = "degraded"
        status["error"] = "OPENAI_API_KEY not set"
    
    return status


# ============================================================
#  SERVE FRONTEND (must be LAST)
# ============================================================
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Startup message
print("=" * 50)
print("Rules Assistant API Starting...")
print(f"  DATA_DIR: {DATA_DIR}")
print(f"  FRONTEND_DIR: {FRONTEND_DIR}")
print(f"  OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
print("=" * 50)
