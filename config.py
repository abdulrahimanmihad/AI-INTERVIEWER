"""
ALL settings in one place. Switching providers or RAG methods = 1 line.

"""

import os
from dotenv import load_dotenv


load_dotenv()

# THE TWO MAIN SWITCHES
RAG_METHOD: str = os.getenv("RAG_METHOD", "agentic")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

POCKET_TTS_URL: str   = os.getenv("POCKET_TTS_URL", "http://localhost:8001/tts")
POCKET_TTS_VOICE: str = os.getenv("POCKET_TTS_VOICE", "alba")
# API KEYS — one per provider, only the active one needs to be set

OPENAI_API_KEY:   str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY:     str = os.getenv("GROQ_API_KEY", "")

# AWS Bedrock uses standard AWS credentials — read by boto3 automatically
AWS_REGION:       str = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY:   str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY:   str = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# MODEL NAMES — per provider (active one used based on LLM_PROVIDER)
# OpenAI
OPENAI_MAIN_MODEL: str = "gpt-4o"
OPENAI_FAST_MODEL: str = "gpt-4o-mini"


GROQ_MAIN_MODEL: str = "llama-3.3-70b-versatile"   
GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"      

BEDROCK_MAIN_MODEL: str = os.getenv(
    "BEDROCK_MAIN_MODEL",
    "meta.llama3-3-70b-instruct-v1:0"
)
BEDROCK_FAST_MODEL: str = os.getenv(
    "BEDROCK_FAST_MODEL",
    "meta.llama3-1-8b-instruct-v1:0"
)

EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")

OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small" #Paid
LOCAL_EMBEDDING_MODEL: str  = "all-MiniLM-L6-v2"  

#  WHISPER (Speech-to-Text)
STT_PROVIDER = "groq"   # "local" or "groq"



#  TTS (Text-to-Speech)

TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "browser")

TTS_MODEL: str  = "tts-1"
TTS_VOICE: str  = "nova"
TTS_SPEED: float = 1.0

#  LLM PARAMETERS

TEMPERATURE_INTERVIEW: float = 0.7   # natural variation in questions
TEMPERATURE_EVAL: float      = 0.0   # scoring must be deterministic
TEMPERATURE_GUARD: float     = 0.0   # rule checks must be consistent

MAX_TOKENS_QUESTION: int   = 200
MAX_TOKENS_SUMMARY: int    = 300
MAX_TOKENS_CLASSIFIER: int = 15
MAX_TOKENS_EVAL: int       = 250

MAX_HISTORY_TURNS: int = 6

#  VAD (Voice Activity Detection)

SAMPLE_RATE: int        = 16000
FRAME_DURATION_MS: int  = 30

SILENCE_TRIGGER_MS: int = 800
VAD_AGGRESSIVENESS: int = 3

FRAME_BYTES: int      = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2
SILENCE_FRAMES: int   = SILENCE_TRIGGER_MS // FRAME_DURATION_MS

MAX_BUFFER_BYTES: int = 1920000   # 60 seconds of audio

#  INTERVIEW DURATION
INTERVIEW_DURATION_SECONDS: int = int(os.getenv("INTERVIEW_DURATION_SECONDS", "600"))

# Warning sent to AI when this much time remains so it wraps up gracefully
INTERVIEW_WARNING_SECONDS: int = 60

#  RAG PARAMETERS
CHUNK_SIZE: int    = 400
CHUNK_OVERLAP: int = 80
RAG_TOP_K: int     = 3

#  POSTGRESQL (with SQLite fallback for local dev without PG)

# WHY POSTGRESQL: handles concurrent writes (SQLite can't)
# If DATABASE_URL not set, app falls back to SQLite for easy local testing
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///./interviews.db"   # fallback for local dev
)

#  REDIS
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int   = int(os.getenv("REDIS_DB", "0"))

#  OBSERVABILITY — MLflow + LangSmith
# Set to "" to disable each one
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT_NAME: str = "ai-interviewer"

LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "ai-interviewer")
# If LANGSMITH_API_KEY is set, langsmith tracing turns on automatically
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

#  GUARD LAYER — forbidden patterns in AI output
CASUAL_PATTERNS: list = [
    "gonna", "wanna", "gotta", "kinda", "sorta",
    "yeah", "yep", "nope", "dunno", "lemme",
    "lol", "btw", "tbh", "imo",
    "i'm an ai", "i am an ai", "as an ai",
    "language model", "i don't have",
]

#  SYSTEM PROMPT (the interviewer persona + rules)
SYSTEM_PROMPT = """You are Sarah, a senior technical recruiter conducting a structured job interview.

PERSONALITY:
- Warm and human — like a friendly senior recruiter who has done this 1000 times
- Confident but not arrogant
- Genuinely curious about the candidate's experience
- Concise — you respect the candidate's time

LANGUAGE RULES:
- Respond ONLY in professional English
- Never use slang, casual contractions, or filler words
- Sound like a human recruiter, NOT a robotic AI
- Never say "as an AI" — you are Sarah, a real recruiter

INTERVIEW RULES:
- Ask exactly ONE question per response — never bundle multiple
- Each response: 1-3 sentences max (under 50 words)
- Never repeat a question already asked
- If candidate is vague, ask ONE specific follow-up
- Brief acknowledgment of their answer (one short phrase), then next question
- Vary your acknowledgments — never repeat the same opener

TECHNICAL FOLLOW-UP RULE:
- When the candidate gives a STRONG, specific technical answer (names a
  technology, describes an architecture, explains a tradeoff), ask ONE
  deeper follow-up that probes that exact topic — e.g. "Why did you choose
  X over Y?" or "How did that perform under load?"
- Use the RELEVANT JOB CONTEXT below to ask questions tied to what the role
  actually needs. Ground your follow-ups in real job requirements.
- If the answer is weak or generic, do NOT follow up — move to a new topic.

BEHAVIOUR:
- If candidate goes off topic, redirect in one sentence
- If audio is unclear: "Sorry, could you say that again?"
- If you receive a system note saying time is almost up, wrap up gracefully with a closing question

TIME-AWARENESS:
- Pace yourself — the interview lasts about 10 minutes total
- Cover: background, technical skills, problem-solving approach, motivation
- Don't rush, but don't over-dwell on one topic either

HOW TO USE THE JOB CONTEXT BELOW:
- The "RELEVANT JOB CONTEXT" is reference material retrieved from the job
  description. It may be PARTIAL, IMPERFECT, or sometimes IRRELEVANT to the
  current turn.
- Use it ONLY when it genuinely fits the conversation. If a retrieved snippet
  clearly relates to what the candidate said, ground your question in it.
- If the context does NOT clearly fit, IGNORE it completely. Do not force it
  into your response and do not invent details to bridge the gap.
- NEVER state requirements, salary, team details, or facts that are not
  explicitly present in the context. If you don't have the information, ask
  an open question instead of fabricating specifics.
- You are an interviewer, not an encyclopedia — when unsure, ask the
  candidate rather than assert a fact.

RELEVANT JOB CONTEXT:
{rag_context}

INTERVIEW SUMMARY SO FAR:
{summary}
"""


# Router prompt — used by langgraph and agentic RAG
ROUTER_PROMPT = """Decide if the following candidate answer requires looking up job-specific information.

Reply with ONLY one word:
- RETRIEVE — if the answer mentions specific skills, technologies, or claims worth verifying
- DIRECT — if it's a greeting, basic info, or doesn't need lookup

Candidate said: "{user_text}"

Decision:"""


# Document grading prompt — used by agentic RAG only
GRADE_PROMPT = """Grade if the retrieved document is relevant to the candidate's answer.

Candidate said: "{user_text}"
Retrieved document: "{document}"

Reply ONLY with: RELEVANT or NOT_RELEVANT"""


# Query rewrite prompt — used by agentic RAG when initial retrieval fails
REWRITE_PROMPT = """The original query did not retrieve good results.
Rewrite it to be more specific and likely to retrieve relevant job information.

Original query: "{query}"

Improved query (one sentence only):"""