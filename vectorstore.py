import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from embeddings import embeddings  # the shared embedding model


#  SHARED VECTOR STORE
vectorstore = Chroma(
    collection_name="interview_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


#  TEXT SPLITTER — shared too
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def ingest_document(text: str, doc_id: str) -> int:
    """
    Store a document in the shared vector DB.
    All three RAG methods can search what we store here.
    """
    if not text or not text.strip():
        return 0

    chunks = _splitter.split_text(text)
    vectorstore.add_texts(
        texts=chunks,
        metadatas=[{"doc_id": doc_id, "i": i} for i in range(len(chunks))],
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
    )
    print(f"[VECTORSTORE] Stored '{doc_id}': {len(chunks)} chunks")
    return len(chunks)


def load_knowledge_base(directory: str = "./docs") -> int:
    """Load all .txt files from a folder."""
    if not os.path.exists(directory):
        print(f"[VECTORSTORE] '{directory}' not found")
        return 0

    loaded = 0
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        path   = os.path.join(directory, filename)
        doc_id = filename.replace(".txt", "")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            ingest_document(text, doc_id)
            loaded += 1
    print(f"[VECTORSTORE] Loaded {loaded} document(s)")
    return loaded


def count() -> int:
    """How many chunks are in the DB? Useful before querying."""
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0
