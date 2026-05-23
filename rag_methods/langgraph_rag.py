"""
rag_methods/langgraph_rag.py

LANGGRAPH RAG workflow as a graph with conditional retrieval.

DIFFERENT FROM CLASSIC:
    Classic: retrieves EVERY turn (sometimes wasteful)
    LangGraph: skips retrieval for greetings, fillers, "thanks"

DIFFERENT FROM AGENTIC:
    LangGraph: simple yes/no routing
    Agentic: also grades retrieved docs, rewrites query if bad

USE CASE:
    Good middle ground — smarter than classic, simpler than agentic
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from config import (
    SYSTEM_PROMPT,
    ROUTER_PROMPT,
    CASUAL_PATTERNS,
    MAX_TOKENS_QUESTION,
    MAX_TOKENS_CLASSIFIER,
    TEMPERATURE_INTERVIEW,
    TEMPERATURE_GUARD,
    RAG_TOP_K,
)
from llm_provider import llm_chat, get_fast_model
from vectorstore import vectorstore, count



# STATE — what flows between nodes

class GraphState(TypedDict):
    user_text:      str
    history:        List[dict]
    summary:        str
    rag_context:    str
    response:       str
    tokens_used:    int
    interview_done: bool
    time_remaining: int   
    _route:         str   



# NODE 1 — ROUTE (do we need RAG?)

async def route_node(state: GraphState) -> GraphState:
    """
    Decide RETRIEVE vs DIRECT using a fast, cheap LLM classification.

    RULE:
        RETRIEVE — candidate discusses skills/tech/the role, or asks about
                   the job/company → we want job-doc context
        DIRECT   — greetings, acknowledgments, small talk → skip retrieval
    """
    prompt = ROUTER_PROMPT.format(user_text=state["user_text"][:300])

    text, tokens = await llm_chat(
        messages=[{"role": "user", "content": prompt}],
        model=get_fast_model(),
        max_tokens=MAX_TOKENS_CLASSIFIER,
        temperature=0,
    )

    state["_route"]       = "DIRECT" if "DIRECT" in text.upper() else "RETRIEVE"
    state["tokens_used"] += tokens
    return state


def should_retrieve(state: GraphState) -> str:
    """Edge function picks next node based on route_node's decision."""
    return state["_route"]



#  NODE 2 RETRIEVE

async def retrieve_node(state: GraphState) -> GraphState:
    """Search the vector DB."""
    if count() == 0:
        state["rag_context"] = ""
        return state

    results = vectorstore.similarity_search(
        query=state["user_text"],
        k=min(RAG_TOP_K, count()),
    )

    if results:
        lines = [f"[{i+1}] {doc.page_content.strip()}" for i, doc in enumerate(results)]
        state["rag_context"] = "Relevant job information:\n" + "\n".join(lines)
    else:
        state["rag_context"] = ""

    return state



#  NODE 2  SKIP RETRIEVE (set empty context)

async def skip_retrieve_node(state: GraphState) -> GraphState:
    state["rag_context"] = ""
    return state



#  NODE 3 — GENERATE

async def generate_node(state: GraphState) -> GraphState:
    """Generate the response using context + history + time awareness."""
    # ── TIME-AWARE INSTRUCTION ──
    # Tell the LLM how much time is left so it can pace the interview:
    #   > 3 min: ask normal questions + follow-ups
    #   last 2 min: invite the candidate to ask THEIR questions
    #   under 1 min: wrap up warmly
    secs = state.get("time_remaining", 9999)
    mins_left = secs // 60
    secs_left = secs % 60

    if secs <= 60:
        time_note = (f"Only {secs} seconds remain. Give a brief warm closing "
                     "statement and thank the candidate. Do not ask new questions.")
    elif secs <= 120:
        time_note = (f"About {mins_left}m {secs_left}s remain. Wind down — invite "
                     "the candidate to ask YOU any questions about the role or team.")
    elif secs <= 180:
        time_note = (f"About {mins_left}m {secs_left}s remain. Start moving toward "
                     "closing topics; ask one final substantive question.")
    else:
        time_note = (f"About {mins_left} minutes remain. Ask focused questions and, "
                     "when the candidate gives a strong technical answer, ask a "
                     "follow-up that probes deeper into that specific area.")

    system_content = SYSTEM_PROMPT.format(
        rag_context=state["rag_context"] or "(No specific context.)",
        summary=state["summary"] or "(Interview just started.)",
    )

    system_content += f"\n\nTIME GUIDANCE: {time_note}"

    messages = [{"role": "system", "content": system_content}] + state["history"]

    text, tokens = await llm_chat(
        messages=messages,
        max_tokens=MAX_TOKENS_QUESTION,
        temperature=TEMPERATURE_INTERVIEW,
    )

    state["interview_done"] = "[INTERVIEW_DONE]" in text
    state["response"]       = text.replace("[INTERVIEW_DONE]", "").strip()
    state["tokens_used"]   += tokens
    return state


#  NODE 4  GUARD

async def guard_node(state: GraphState) -> GraphState:
    """Check for casual language, regenerate once if found."""
    response_lower = state["response"].lower()
    if not any(p in response_lower for p in CASUAL_PATTERNS):
        return state

    print("[LANGGRAPH RAG] Casual detected — regenerating")

    system_content = SYSTEM_PROMPT.format(
        rag_context=state["rag_context"] or "(No context.)",
        summary=state["summary"] or "(Start.)",
    )
    messages = (
        [{"role": "system", "content": system_content}]
        + state["history"]
        + [{"role": "assistant", "content": state["response"]}]
        + [{"role": "user", "content": "Rephrase that formally as a senior recruiter."}]
    )

    text, tokens = await llm_chat(
        messages=messages,
        max_tokens=MAX_TOKENS_QUESTION,
        temperature=TEMPERATURE_GUARD,
    )
    state["response"]     = text
    state["tokens_used"] += tokens
    return state


#  BUILD GRAPH (once at module load)

def _build_graph():
    g = StateGraph(GraphState)
    g.add_node("route",         route_node)
    g.add_node("retrieve",      retrieve_node)
    g.add_node("skip_retrieve", skip_retrieve_node)
    g.add_node("generate",      generate_node)
    g.add_node("guard",         guard_node)

    g.set_entry_point("route")
    g.add_conditional_edges("route", should_retrieve, {
        "RETRIEVE": "retrieve",
        "DIRECT":   "skip_retrieve",
    })
    g.add_edge("retrieve",      "generate")
    g.add_edge("skip_retrieve", "generate")
    g.add_edge("generate",      "guard")
    g.add_edge("guard",         END)

    return g.compile()


_graph = _build_graph()



#  PUBLIC API

async def run_turn(user_text: str, history: List[dict], summary: str,
                   time_remaining: int = 9999) -> dict:
    """Same signature as the other two methods — they're swappable."""
    initial_state: GraphState = {
        "user_text":      user_text,
        "history":        history,
        "summary":        summary,
        "rag_context":    "",
        "response":       "",
        "tokens_used":    0,
        "interview_done": False,
        "time_remaining": time_remaining,
        "_route":         "",
    }

    final = await _graph.ainvoke(initial_state)

    return {
        "response":       final["response"],
        "tokens_used":    final["tokens_used"],
        "interview_done": final["interview_done"],
        "rag_used":       bool(final["rag_context"]),
        "method":         "langgraph",
    }