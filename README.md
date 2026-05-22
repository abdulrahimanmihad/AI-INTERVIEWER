# AI Interviewer — Switchable RAG, LLM, and TTS

Voice-based AI interviewer with three swappable RAG methods, three LLM providers, MLflow + LangSmith observability, PostgreSQL persistence, and a free-tier-only setup option.

## What can be switched (with one config change)

| What | Options |
|---|---|
| **RAG method** | `classic` / `langgraph` / `agentic` |
| **LLM provider** | `groq` (free) / `openai` (paid) / `bedrock` (AWS) |
| **Embeddings** | `local` (free) / `openai` (paid) |
| **TTS** | `browser` (free) / `openai` (paid) |
| **Database** | SQLite (local) / PostgreSQL (production) |

Change them in `.env` — restart server — done. No code edits.

## Project structure

```
interviewer/
├── config.py              ← all switches and settings
├── llm_provider.py        ← unified LLM interface (OpenAI/Groq/Bedrock)
├── embeddings.py          ← shared embedding model (local or OpenAI)
├── vectorstore.py         ← shared ChromaDB instance
├── database.py            ← PostgreSQL/SQLite via SQLAlchemy
├── observability.py       ← MLflow + LangSmith tracking
├── rag_methods/
│   ├── classic_rag.py     ← always retrieves, then generates
│   ├── langgraph_rag.py   ← LangGraph with conditional retrieval
│   └── agentic_rag.py     ← full agent: route + grade + rewrite + generate
├── rag_factory.py         ← picks ONE method based on config.RAG_METHOD
├── main.py                ← FastAPI app, WebSocket, STT, VAD
├── static/index.html      ← frontend (browser TTS, free)
├── docs/                  ← .txt files = your RAG knowledge base
├── requirements.txt
├── Dockerfile
├── docker-compose.yml     ← one command to start everything
└── .env.example
```

## Free-tier setup (no credit card)

This setup runs entirely on free services:
- **LLM** → Groq (free tier, Llama 3.3 70B, very fast)
- **STT** → local Whisper (free, runs on your laptop)
- **TTS** → browser speech synthesis (free, no API)
- **Embeddings** → sentence-transformers local model (free)
- **MLflow** → local file storage (free)
- **LangSmith** → free tier 5000 traces/month (optional)

### Get the free Groq key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no card needed)
3. Create an API key

### Run it

```bash
# 1. Clone
cd interviewer

# 2. Create virtual env
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY
# Leave RAG_METHOD=agentic, LLM_PROVIDER=groq, TTS_PROVIDER=browser

# 5. Start Redis (one-time, in a separate terminal)
docker run -d --name redis -p 6379:6379 redis

# 6. Run
uvicorn main:app --reload

# 7. Open http://localhost:8000
```

## With Docker Compose (recommended — starts Redis + Postgres + app)

```bash
# Set your API key as an environment variable first
export GROQ_API_KEY=your-key

# Start everything
docker-compose up

# Open http://localhost:8000
```

## How to compare the three RAG methods

The whole point of having three methods is to find out which works best for **your** data.

### Step 1 — Run interviews with each method

```bash
# Try classic
echo "RAG_METHOD=classic" >> .env
uvicorn main:app --reload
# Do 5 interviews

# Try langgraph
sed -i 's/RAG_METHOD=classic/RAG_METHOD=langgraph/' .env
# Restart server, do 5 more interviews

# Try agentic
sed -i 's/RAG_METHOD=langgraph/RAG_METHOD=agentic/' .env
# Restart server, do 5 more interviews
```

### Step 2 — View results in MLflow

```bash
mlflow ui
# Open http://localhost:5000
```

You'll see all interviews with these comparable metrics:

| Metric | What it tells you |
|---|---|
| `latency_seconds` | How fast each turn was |
| `tokens` | API cost per turn |
| `rag_used` | Whether retrieval happened |
| `total_turns` | How long interviews lasted |
| `success` | Did the turn complete without crashing |

### Step 3 — Pick the winner

Sort by `latency_seconds` and `tokens`. Best balance wins.

Rough expectations:
- **Classic** — fastest, but spends tokens on every turn
- **LangGraph** — saves ~40% tokens by skipping retrieval on greetings
- **Agentic** — best quality answers, ~2x slower, uses more tokens

## How to view LLM traces in LangSmith

```bash
# Get a key from https://smith.langchain.com (free tier)
# Add to .env:
LANGSMITH_API_KEY=ls-...

# Restart server
# Every LLM call now auto-traces to LangSmith
# View at https://smith.langchain.com/projects/ai-interviewer
```

You'll see every LLM call, every prompt, every response — invaluable for debugging.

## Deploy

### Railway (easiest, free tier)

1. Push code to GitHub
2. Railway → New Project → Deploy from GitHub
3. Add Redis service (one click)
4. Add Postgres service (one click)
5. Set environment variables (`GROQ_API_KEY` etc.)
6. Railway auto-detects Dockerfile and deploys

### Fly.io

```bash
fly launch
fly secrets set GROQ_API_KEY=your-key
fly deploy
```

## Architecture diagram

```
                      Browser
                         │
                  audio over WebSocket
                         │
                         ↓
                  ┌─────────────┐
                  │  main.py    │
                  └──────┬──────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ↓                                 ↓
  vad_receiver_loop              turn_processor_loop
  (always receiving)               (processes turns)
        │                                 │
        └─── audio_queue ────────────────→│
                                          ↓
                                    1. Whisper STT
                                          │
                                          ↓
                                    2. rag_factory.run_turn()
                                       ├── classic_rag.py
                                       ├── langgraph_rag.py
                                       └── agentic_rag.py
                                          │
                                          ↓
                                    3. Send text → browser TTS
                                          │
                                          ↓
                                    4. MLflow logs metrics
                                       LangSmith logs LLM calls
                                          │
                                          ↓
                                    5. Save state to Redis
                                          │
                                          ↓
                              (when done) Archive to PostgreSQL
```

## Key teaching points (for explaining to him)

- **rag_factory.py** is the Strategy pattern — main.py doesn't know which RAG method runs.
- **llm_provider.py** is the Adapter pattern — every other file uses the same interface regardless of provider.
- **vectorstore.py** is the single source for embeddings — impossible to use a different model for ingest vs query.
- **Module-level Whisper load** — the most important latency optimization in the entire app.
- **VAD/processor decoupling** — audio is never dropped while STT/LLM runs.
- **SQL-then-Redis-delete ordering** — data is never lost on archive failure.

Each design decision is documented in inline comments in each file.
