import os
import sqlite3
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
import httpx

# ============================================================
#  OPENAI KEY (from Railway environment variable)
# ============================================================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set. Add it in Railway Variables.")

def get_openai():
    http_client = httpx.Client(
        timeout=60.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=40)
    )
    return OpenAI(api_key=openai_api_key, http_client=http_client)

client = get_openai()

# ============================================================
#  DATA PATHS (Railway structure)
# ============================================================
BASE_DIR = Path(__file__).parent.parent  # Go up from backend/ to root
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

DB_PATH = DATA_DIR / "chunks.db"
INDEX_PATH = DATA_DIR / "rulebook.index"
SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# Load FAISS index
index = faiss.read_index(str(INDEX_PATH))

# Connect to SQLite
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
conn.row_factory = sqlite3.Row

# Load system prompt (or use default)
if SYSTEM_PROMPT_PATH.exists():
    SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
else:
    SYSTEM_PROMPT = """You are an expert rules assistant for a tabletop game.
You must answer ONLY using the provided rulebook excerpts.

Format:
SHORT ANSWER: [Clear answer with brief mechanical context]

REASONING: [Detailed explanation with citations]

If the excerpts don't contain the answer:
SHORT ANSWER: No explicit rule found.
REASONING: The rulebook sections provided do not cover this specific situation."""

# ============================================================
#  SESSION MEMORY
# ============================================================
last_questions = {}

# ============================================================
#  FASTAPI SETUP
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
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def rewrite_query(user_message: str, session_id: Optional[str]) -> str:
    """Rewrite casual questions into rulebook search terms."""
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
    chunk_data = []
    for faiss_idx in indices:
        sqlite_id = faiss_idx + 1  # FAISS 0-indexed, SQLite 1-indexed
        row = conn.execute("SELECT page, content FROM chunks WHERE id=?", (sqlite_id,)).fetchone()
        if row:
            chunk_data.append({"page": row["page"], "content": row["content"]})
    return chunk_data

def generate_answer(question: str, rewritten: str, chunks: List[Dict]) -> str:
    combined_text = "\n\n".join(f"[p{c['page']}] {c['content']}" for c in chunks)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    return {"status": "healthy", "chunks": index.ntotal}

# ============================================================
#  SERVE FRONTEND (Railway serves static files)
# ============================================================
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

print("Rules Assistant API ready!")
