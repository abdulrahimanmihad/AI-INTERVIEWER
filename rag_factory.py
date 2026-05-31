from config import RAG_METHOD


if RAG_METHOD == "prebuilt":
    from rag_methods.prebuilt_rag import run_turn 
    print(f"[RAG FACTORY] Using prebuilt RAG")
elif RAG_METHOD == "classic":
    from rag_methods.classic_rag import run_turn
    print(f"[RAG FACTORY] Using CLASSIC RAG")

elif RAG_METHOD == "langgraph":
    from rag_methods.langgraph_rag import run_turn
    print(f"[RAG FACTORY] Using LANGGRAPH RAG")

elif RAG_METHOD == "agentic":
    from rag_methods.agentic_rag import run_turn
    print(f"[RAG FACTORY] Using AGENTIC RAG")

else:
    raise ValueError(
        f"Unknown RAG_METHOD: {RAG_METHOD}. "
        f"Must be 'classic', 'langgraph', or 'agentic'."
    )


# Re-export run_turn so main.py can import it without knowing the method
__all__ = ["run_turn"]
