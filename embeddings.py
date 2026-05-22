"""
embeddings.py
═════════════════════════════════════════════════════════════════════
Embedding model wrapper — same as llm_provider.py but for embeddings.

CRITICAL RULE: the SAME embedding model must be used for storing AND searching.
This file is the single place where the embedding model is created.
All RAG files import _embeddings from here.
That makes mismatches IMPOSSIBLE.

PROVIDERS:
    "openai" → text-embedding-3-small (paid, 1536 dims, very good)
    "local"  → all-MiniLM-L6-v2       (free, 384 dims, runs offline)
═════════════════════════════════════════════════════════════════════
"""

from config import (
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
)


# Build the LangChain-compatible embedding object based on config
if EMBEDDING_PROVIDER == "openai":
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
    print(f"[EMBED] Using OpenAI embeddings ({OPENAI_EMBEDDING_MODEL})")

elif EMBEDDING_PROVIDER == "local":
    # HuggingFaceEmbeddings runs sentence-transformers locally
    # First run: downloads ~90MB model. After that: instant, offline, free.
    # WHY THIS IS GREAT FOR FREE TIER:
    #   Zero API calls = zero cost
    #   Works without internet after first download
    #   Quality is good enough for English RAG
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[EMBED] Using local embeddings ({LOCAL_EMBEDDING_MODEL}) — FREE")

else:
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
