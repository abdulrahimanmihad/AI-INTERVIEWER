"""
rag_factory.py
═════════════════════════════════════════════════════════════════════
The "Factory" — picks which RAG method to use based on config.RAG_METHOD.

WHY A FACTORY:
    main.py imports run_turn from here.
    main.py NEVER imports specific RAG methods.
    Change config.RAG_METHOD → factory picks the new one → main.py unchanged.

THIS IS HOW YOU SWITCH RAG METHODS:
    In .env or environment: set RAG_METHOD=classic (or langgraph, or agentic)
    Restart server.
    That's it.
═════════════════════════════════════════════════════════════════════
"""

from config import RAG_METHOD


# Import the chosen RAG method's run_turn function
# WHY IMPORT INSIDE THE IF:
#   We only want to load (and pay the import cost of) the active method
#   Importing langgraph_rag also imports langgraph itself — slow
#   With this pattern, only the active method's code is loaded

if RAG_METHOD == "classic":
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
