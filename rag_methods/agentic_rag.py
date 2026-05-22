"""
rag_methods/agentic_rag.py

AGENTIC RAG full agent with self-correction.


THIS IS "AGENTIC" BECAUSE:
    The agent makes DECISIONS at multiple steps:
    1. Should I retrieve?
    2. Are the retrieved docs actually useful? (grade)
    3. If not, should I rewrite my query?
    4. Should I generate or try again?

WHY THIS IS BETTER THAN PLAIN LANGGRAPH:
    LangGraph just decides "retrieve or not"
    Agentic also catches BAD retrievals and rewrites the query
    Result: higher quality context, fewer hallucinations
    Cost: more API calls per turn

TRADEOFF:
    Best quality, slowest, most expensive
    Use when accuracy matters more than latency

"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from config import (
    SYSTEM_PROMPT,
    ROUTER_PROMPT,
    GRADE_PROMPT,
    REWRITE_PROMPT,
    CASUAL_PATTERNS,
    MAX_TOKENS_QUESTION,
    MAX_TOKENS_CLASSIFIER,
    TEMPERATURE_INTERVIEW,
    TEMPERATURE_GUARD,
    RAG_TOP_K,
)
from llm_provider import llm_chat, get_fast_model
from vectorstore import vectorstore, count


#  STATE

class AgentState(TypedDict):
    user_text:      str
    query:          str   # may be rewritten by rewrite_node
    history:        List[dict]
    summary:        str
    rag_context:    str
    rag_docs:       List[str]   # raw chunks before formatting
    rewrite_count:  int          # prevent infinite rewrites
    response:       str
    tokens_used:    int
    interview_done: bool
    time_remaining: int   
    _route:         str
    _grade:         str



#  NODE 1 — ROUTE

async def route_node(state: AgentState) -> AgentState:
    prompt = ROUTER_PROMPT.format(user_text=state["user_text"][:300])

    text, tokens = await llm_chat(
        messages=[{"role": "user", "content": prompt}],
        model=get_fast_model(),
        max_tokens=MAX_TOKENS_CLASSIFIER,
        temperature=0,
    )

    state["_route"]       = "RETRIEVE" if "RETRIEVE" in text.upper() else "DIRECT"
    state["tokens_used"] += tokens
    return state


def should_retrieve(state: AgentState) -> str:
    return state["_route"]



#  NODE 2 — RETRIEVE

async def retrieve_node(state: AgentState) -> AgentState:
    """Search using state['query'] (may be rewritten version)."""
    query = state["query"] or state["user_text"]

    if count() == 0:
        state["rag_docs"]    = []
        state["rag_context"] = ""
        return state

    results = vectorstore.similarity_search(
        query=query,
        k=min(RAG_TOP_K, count()),
    )

    state["rag_docs"] = [doc.page_content.strip() for doc in results]
    return state


#  NODE 3 — GRADE retrieved documents

async def grade_node(state: AgentState) -> AgentState:
    """
    Ask the LLM: are these docs actually relevant to the candidate's answer?

    This is the "self-correction" part — the agent doesn't trust the
    vector search blindly. It double-checks.

    WHY THIS MATTERS:
        Vector search finds the most SIMILAR chunks — but similar isn't
        always RELEVANT. The candidate might mention something we have
        no docs about. Grading catches that case.
    """
    if not state["rag_docs"]:
        state["_grade"] = "NOT_RELEVANT"
        return state

    relevant_docs = []
    for doc in state["rag_docs"]:
        prompt = GRADE_PROMPT.format(
            user_text=state["user_text"][:200],
            document=doc[:300],
        )
        text, tokens = await llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model=get_fast_model(),
            max_tokens=MAX_TOKENS_CLASSIFIER,
            temperature=0,
        )
        state["tokens_used"] += tokens

        if "RELEVANT" in text.upper() and "NOT" not in text.upper():
            relevant_docs.append(doc)

    if relevant_docs:
        lines = [f"[{i+1}] {d}" for i, d in enumerate(relevant_docs)]
        state["rag_context"] = "Relevant job information:\n" + "\n".join(lines)
        state["_grade"]      = "RELEVANT"
    else:
        state["rag_context"] = ""
        state["_grade"]      = "NOT_RELEVANT"

    return state


def should_rewrite(state: AgentState) -> str:
    """
    If docs were not relevant AND we haven't rewritten before → try rewriting query
    Otherwise → just proceed to generate (with empty or whatever context we have)
    """
    if state["_grade"] == "NOT_RELEVANT" and state["rewrite_count"] < 1:
        return "REWRITE"
    return "GENERATE"



#  NODE 4 — REWRITE QUERY (if grading failed)

async def rewrite_node(state: AgentState) -> AgentState:
    """
    Ask the LLM to rewrite the query to be more specific.
    Then we go back to retrieve_node with the new query.
    """
    prompt = REWRITE_PROMPT.format(query=state["query"] or state["user_text"])

    text, tokens = await llm_chat(
        messages=[{"role": "user", "content": prompt}],
        model=get_fast_model(),
        max_tokens=100,
        temperature=0.3,
    )

    state["query"]          = text.strip().strip('"').strip("'")
    state["rewrite_count"] += 1
    state["tokens_used"]   += tokens
    print(f"[AGENTIC] Rewrote query → '{state['query'][:80]}'")
    return state

#  NODE — SKIP RETRIEVE

async def skip_retrieve_node(state: AgentState) -> AgentState:
    state["rag_context"] = ""
    return state


#  NODE 5 — GENERATE

async def generate_node(state: AgentState) -> AgentState:
    system_content = SYSTEM_PROMPT.format(
        rag_context=state["rag_context"] or "(No specific context.)",
        summary=state["summary"] or "(Interview just started.)",
    )
    
    system_content += "\n\nTIME GUIDANCE: " + _time_note(state.get("time_remaining", 9999))

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


def _time_note(secs: int) -> str:

    mins_left = secs // 60
    secs_left = secs % 60
    if secs <= 60:
        return (f"Only {secs} seconds remain. Give a brief warm closing "
                "statement and thank the candidate. Do not ask new questions.")
    elif secs <= 120:
        return (f"About {mins_left}m {secs_left}s remain. Wind down — invite "
                "the candidate to ask YOU any questions about the role or team.")
    elif secs <= 180:
        return (f"About {mins_left}m {secs_left}s remain. Start moving toward "
                "closing topics; ask one final substantive question.")
    else:
        return (f"About {mins_left} minutes remain. Ask focused questions and, "
                "when the candidate gives a strong technical answer, ask a "
                "follow-up that probes deeper into that specific area.")



#  NODE 6 — GUARD
async def guard_node(state: AgentState) -> AgentState:
    response_lower = state["response"].lower()
    if not any(p in response_lower for p in CASUAL_PATTERNS):
        return state

    print("[AGENTIC RAG] Casual detected — regenerating")

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



#  BUILD GRAPH

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("route",         route_node)
    g.add_node("retrieve",      retrieve_node)
    g.add_node("grade",         grade_node)
    g.add_node("rewrite",       rewrite_node)
    g.add_node("skip_retrieve", skip_retrieve_node)
    g.add_node("generate",      generate_node)
    g.add_node("guard",         guard_node)

    g.set_entry_point("route")

    g.add_conditional_edges("route", should_retrieve, {
        "RETRIEVE": "retrieve",
        "DIRECT":   "skip_retrieve",
    })

    g.add_edge("retrieve", "grade")

    g.add_conditional_edges("grade", should_rewrite, {
        "REWRITE":  "rewrite",
        "GENERATE": "generate",
    })

    # After rewrite, go back to retrieve with the new query
    g.add_edge("rewrite", "retrieve")

    g.add_edge("skip_retrieve", "generate")
    g.add_edge("generate",      "guard")
    g.add_edge("guard",         END)

    return g.compile()


_graph = _build_graph()



#  PUBLIC API 

async def run_turn(user_text: str, history: List[dict], summary: str, time_remaining: int = 9999) -> dict:
    initial_state: AgentState = {
        "user_text":      user_text,
        "query":          user_text,    # initial query = user text
        "history":        history,
        "summary":        summary,
        "rag_context":    "",
        "rag_docs":       [],
        "rewrite_count":  0,
        "response":       "",
        "tokens_used":    0,
        "interview_done": False,
        "time_remaining": time_remaining,
        "_route":         "",
        "_grade":         "",
    }

    final = await _graph.ainvoke(initial_state)

    return {
        "response":       final["response"],
        "tokens_used":    final["tokens_used"],
        "interview_done": final["interview_done"],
        "rag_used":       bool(final["rag_context"]),
        "method":         "agentic",
    }