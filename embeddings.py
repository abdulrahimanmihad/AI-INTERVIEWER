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
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[EMBED] Using local embeddings ({LOCAL_EMBEDDING_MODEL}) — FREE")

else:
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
