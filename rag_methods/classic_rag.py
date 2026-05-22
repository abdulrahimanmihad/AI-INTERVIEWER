
from typing import List

from config import (
    SYSTEM_PROMPT,
    MAX_TOKENS_QUESTION,
    TEMPERATURE_INTERVIEW,
    RAG_TOP_K,
    CASUAL_PATTERNS,
    TEMPERATURE_GUARD,
)
from llm_provider import llm_chat
from vectorstore import vectorstore, count


# Always retrieve from vector DB

def _retrieve(query: str) -> str:
    """
    Search vector DB for chunks relevant to query.
    Returns formatted string or empty if DB has nothing.
    """
    if count() == 0:
        return ""

    results = vectorstore.similarity_search(
        query=query,
        k=min(RAG_TOP_K, count()),
    )

    if not results:
        return ""

    lines = [f"[{i+1}] {doc.page_content.strip()}" for i, doc in enumerate(results)]
    return "Relevant job information:\n" + "\n".join(lines)



# Generate response with retrieved context
async def _generate(
    user_text: str,
    history: List[dict],
    summary: str,
    rag_context: str,
    time_remaining: int = 9999,
) -> tuple[str, int]:
    """Build messages with RAG context + time awareness, call LLM."""
    system_content = SYSTEM_PROMPT.format(
        rag_context=rag_context or "(No specific context.)",
        summary=summary or "(Interview just started.)",
    )
    system_content += "\n\nTIME GUIDANCE: " + _time_note(time_remaining)

    messages = [{"role": "system", "content": system_content}] + history

    return await llm_chat(
        messages=messages,
        max_tokens=MAX_TOKENS_QUESTION,
        temperature=TEMPERATURE_INTERVIEW,
    )


def _time_note(secs: int) -> str:
    """
    Convert seconds-remaining into a pacing instruction for the LLM.
    Identical across classic / langgraph / agentic so behavior matches
    no matter which RAG method is active.
    """
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


#  Guard against casual language
async def _guard(response: str, history: List[dict], summary: str, rag_context: str) -> tuple[str, int]:
    """If casual language detected, regenerate ONCE with correction."""
    response_lower = response.lower()
    if not any(p in response_lower for p in CASUAL_PATTERNS):
        return response, 0

    print("[CLASSIC RAG] Casual detected — regenerating")

    system_content = SYSTEM_PROMPT.format(
        rag_context=rag_context or "(No context.)",
        summary=summary or "(Start.)",
    )
    messages = (
        [{"role": "system", "content": system_content}]
        + history
        + [{"role": "assistant", "content": response}]
        + [{"role": "user", "content": "Rephrase your last response formally as a senior recruiter would."}]
    )

    text, tokens = await llm_chat(
        messages=messages,
        max_tokens=MAX_TOKENS_QUESTION,
        temperature=TEMPERATURE_GUARD,
    )
    return text, tokens


#  PUBLIC API — main.py calls this

async def run_turn(user_text: str, history: List[dict], summary: str, time_remaining: int = 9999) -> dict:

    # Step 1: ALWAYS retrieve (this is what makes it "classic")
    rag_context = _retrieve(user_text)

    # Step 2: generate (time-aware)
    response, tokens = await _generate(user_text, history, summary, rag_context, time_remaining)

    # Step 3: guard
    response, guard_tokens = await _guard(response, history, summary, rag_context)
    tokens += guard_tokens

    # Check for interview completion marker
    interview_done = "[INTERVIEW_DONE]" in response
    response = response.replace("[INTERVIEW_DONE]", "").strip()

    return {
        "response":       response,
        "tokens_used":    tokens,
        "interview_done": interview_done,
        "rag_used":       bool(rag_context),
        "method":         "classic",
    }