"""
rag_methods/prebuilt_rag.py

PREBUILT RAG mode — Sarah asks ONLY questions from the company's bank in
fixed order. After each answer, she classifies it as STRONG or WEAK:
  • STRONG  → SPECIFIC positive reinforcement that names what the candidate
              did right + asks the canned follow-up (deeper probe)
  • WEAK    → brief warm acknowledgment + moves to the next bank question

After the bank is exhausted she invites the candidate to ask their own
questions (answered from JD context) until they end the interview via the
End button or the 5-minute Q&A cap fires.
"""

import logging
from typing import List, Dict

from config import (
    MAX_TOKENS_QUESTION,
    MAX_TOKENS_CLASSIFIER,
    TEMPERATURE_INTERVIEW,
)
from llm_provider import llm_chat, get_fast_model
from question_bank import get_all_questions
from vectorstore import vectorstore, count as _jd_count

log = logging.getLogger(__name__)


# ─── Prompts ───────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """Judge the candidate's answer to the interviewer's question.

Question: "{question}"
Candidate's answer: "{answer}"

Reply with ONE word only:
  STRONG  — answer is specific, technical, gives a real example, names a
            tool/technique/approach, OR shows clear hands-on experience
  WEAK    — answer is vague, generic, says "I don't know" / "not sure",
            one-line, lacks specifics, or off-topic

Reply:"""


PRAISE_AND_FOLLOWUP_PROMPT = """You are Sarah, a senior recruiter. The candidate just gave a STRONG, specific answer to your previous question. You must respond in two parts:

PART 1 — SPECIFIC positive reinforcement (one short sentence):
  Pick out the specific technique, tool, or approach the candidate named and
  validate WHY it was a good choice. Be concrete, not generic.

  GOOD examples (specific, names what they did):
    • "Using XGBoost there was a great move — it really shines on tabular
       data with mixed feature types."
    • "Choosing F1 over plain accuracy was exactly right given the class
       imbalance you described."
    • "Putting a cache between the API and the DB is exactly the right
       move for that hot-key problem."
    • "Catching the data leakage early like that saved you a lot of pain
       downstream."

  BAD examples (generic, do NOT use):
    • "Great answer." / "Sounds good." / "That's interesting."
    • "Nice explanation." / "Good job."

PART 2 — Ask this follow-up (probe deeper):
  {follow_up}

Total response under 50 words. Sound warm and human, not robotic.

The candidate's answer was: "{answer}"
"""

FOLLOWUP_ANSWER_DONE=""" You are Sarah, a senior recruiter. The candidate just gave a STRONG, specific answer to your previous question. You must respond in two parts:

PART 1 — SPECIFIC positive reinforcement (one short sentence):
  Pick out the specific technique, tool, or approach the candidate named and
  validate WHY it was a good choice. Be concrete, not generic.

  GOOD examples (specific, names what they did):
    • "Using XGBoost there was a great move — it really shines on tabular
       data with mixed feature types."
    • "Choosing F1 over plain accuracy was exactly right given the class
       imbalance you described."
    • "Putting a cache between the API and the DB is exactly the right
       move for that hot-key problem."
    • "Catching the data leakage early like that saved you a lot of pain
       downstream."

  BAD examples (generic, do NOT use):
    • "Great answer." / "Sounds good." / "That's interesting."
    • "Nice explanation." / "Good job."

PART 2 — Ask this follow-up (probe deeper):
  {next_question}

Total response under 50 words. Sound warm and human, not robotic.

The candidate's answer was: "{answer}"
"""

ACKNOWLEDGE_AND_NEXT_PROMPT = """You are Sarah, a senior recruiter. The candidate's answer was brief, vague, or they didn't know. Respond in two parts:

PART 1 — Brief warm acknowledgment (5-10 words). Vary across these:
   • "No worries, that's totally fair."
   • "Got it, thanks."
   • "Understood — let's keep going."
   • "Fair enough."
   • "Thanks for sharing."
   • "All good, that happens."

PART 2 — Ask this next interview question, lightly conversational (do not
change the meaning):
   {next_question}

Total response under 45 words. Do not say 'great answer' if it wasn't.
"""


FIRST_QUESTION_PROMPT = """You are Sarah, a senior recruiter starting an interview. Greet the candidate warmly (under 12 words) and ask this question:

   {next_question}

Keep total response under 35 words. Sound human, not scripted.
"""


QA_PHASE_PROMPT = """You are Sarah, a senior recruiter. The candidate has finished all interview questions and may now ask YOU questions about the role.

If the question's answer IS in the JOB CONTEXT below: answer warmly and concisely (under 60 words).

If the answer is NOT in the JOB CONTEXT (salary, team size, manager, exact benefits, internal tools not in the JD): politely defer without inventing anything. Examples:
   • "Great question — I don't have those specifics in front of me, but the hiring team can follow up with you on that."
   • "Honestly, that's one I'd rather not guess on. The team will get back to you with an accurate answer."

After answering or deferring, ask: "Anything else you'd like to know?"

NEVER fabricate facts. NEVER use [INTERVIEW_DONE] — the candidate ends the interview by clicking the End button.

JOB CONTEXT:
{rag_context}
"""


# ─── Helpers ───────────────────────────────────────────────────────

async def _classify_answer(question: str, answer: str) -> bool:
    """Return True if STRONG, False if WEAK. Fast-model call, ~200ms."""
    a = (answer or "").strip()
    if not a or len(a.split()) < 4:
        return False  # too short to be strong; skip the LLM call

    VAGUE_MARKERS = (
        "i don't know", "i dont know", "no idea",
        "i haven't", "i havent", "never used",
    )
    if any(m in a.lower() for m in VAGUE_MARKERS):
        return False  # explicit don't-know; skip LLM call

    try:
        text, _ = await llm_chat(
            messages=[{
                "role": "user",
                "content": CLASSIFIER_PROMPT.format(question=question, answer=a[:600]),
            }],
            model=get_fast_model(),
            max_tokens=MAX_TOKENS_CLASSIFIER,
            temperature=0.0,
        )
        verdict = (text or "").strip().upper()
        return "STRONG" in verdict
    except Exception as e:
        log.error(f"[PREBUILT] classifier failed: {e}")
        return False  # fail-safe: treat as weak, just move on


def _retrieve_jd_context(query: str, k: int = 3) -> str:
    """Pull relevant JD chunks for candidate's question (Q&A phase only)."""
    if _jd_count() == 0:
        return ""
    try:
        results = vectorstore.similarity_search(query=query, k=min(k, _jd_count()))
        if not results:
            return ""
        lines = [f"[{i+1}] {doc.page_content.strip()}" for i, doc in enumerate(results)]
        return "Relevant job information:\n" + "\n".join(lines)
    except Exception as e:
        log.error(f"[PREBUILT] JD retrieval failed: {e}")
        return ""


# ─── Public API ────────────────────────────────────────────────────

async def run_turn(
    user_text: str,
    history: List[Dict],
    summary: str,
    time_remaining: int = 9999,
    *,
    company_id: str = "default",
    asked_question_ids: List[str] = None,
    awaiting_followup: bool = False,
    all_questions_done: bool = False,
) -> Dict:
    """
    Returns:
        response          : Sarah's reply
        tokens_used       : approximate
        interview_done    : False (candidate ends via button)
        rag_used          : True in Q&A phase, else False
        method            : "prebuilt"
        next_question_id  : the question_id "owned" by this turn (for tracking)
        all_questions_done: True if we just exhausted the bank
        followup_asked    : True if we asked a follow-up this turn
    """
    asked_question_ids = asked_question_ids or []
    questions = get_all_questions(company_id)

    if not questions:
        log.error(f"[PREBUILT] No questions for company '{company_id}'")
        return {
            "response":          "I'm sorry — the interview questions aren't available right now. The team will reach out to reschedule.",
            "tokens_used":       0,
            "interview_done":    True,
            "rag_used":          False,
            "method":            "prebuilt",
            "next_question_id":  None,
            "all_questions_done": True,
            "followup_asked":    False,
        }

    # ───── PHASE: Q&A (bank done, candidate asks Sarah) ─────
    if all_questions_done:
        rag_context = _retrieve_jd_context(user_text)
        system_content = QA_PHASE_PROMPT.format(
            rag_context=rag_context or "(No specific JD context retrieved.)",
        )
        messages = [{"role": "system", "content": system_content}] + history[-6:]
        text, tokens = await llm_chat(
            messages=messages,
            max_tokens=MAX_TOKENS_QUESTION,
            temperature=TEMPERATURE_INTERVIEW,
        )
        return {
            "response":          text.strip(),
            "tokens_used":       tokens,
            "interview_done":    False,
            "rag_used":          bool(rag_context),
            "method":            "prebuilt",
            "next_question_id":  None,
            "all_questions_done": True,
            "followup_asked":    False,
        }

    # ───── PHASE: bank questions ─────

    # Special-case the very first question (no prior answer to react to)
    if not asked_question_ids:
        first_q = questions[0]
        system_content = FIRST_QUESTION_PROMPT.format(next_question=first_q["question"])
        text, tokens = await llm_chat(
            messages=[{"role": "system", "content": system_content}],
            max_tokens=MAX_TOKENS_QUESTION,
            temperature=TEMPERATURE_INTERVIEW,
        )
        return {
            "response":          text.strip(),
            "tokens_used":       tokens,
            "interview_done":    False,
            "rag_used":          False,
            "method":            "prebuilt",
            "next_question_id":  first_q["id"],
            "all_questions_done": False,
            "followup_asked":    False,
        }

    # The last bank question we asked (for context: react to candidate's answer)
    last_qid = asked_question_ids[-1]
    last_q   = next((q for q in questions if q["id"] == last_qid), None)

    # If we JUST asked a follow-up and now they answered it, move to next bank
    # question (no further follow-up on the follow-up).
    if awaiting_followup:
        next_q = next((q for q in questions if q["id"] not in asked_question_ids), None)
        if next_q is None:
            return _exhausted_response()
        # We acknowledge briefly (without classifying — they already answered the
        # follow-up, the conversation just moves on smoothly).
        is_strong = await _classify_answer(
        question=last_q["question"] if last_q else "",
        answer=user_text,
        )
        if is_strong:
         system_content = FOLLOWUP_ANSWER_DONE.format(next_question=next_q["question"], answer=user_text[:400])
         messages = [{"role": "system", "content": system_content}] + history[-4:]
         text, tokens = await llm_chat(
            messages=messages,
            max_tokens=MAX_TOKENS_QUESTION,
            temperature=TEMPERATURE_INTERVIEW,
         )
         return {
            "response":          text.strip(),
            "tokens_used":       tokens,
            "interview_done":    False,
            "rag_used":          False,
            "method":            "prebuilt",
            "next_question_id":  next_q["id"],
            "all_questions_done": False,
            "followup_asked":    False,
         }
        else:
         system_content = ACKNOWLEDGE_AND_NEXT_PROMPT.format(next_question=next_q["question"])
         messages = [{"role": "system", "content": system_content}] + history[-4:]
         text, tokens = await llm_chat(
            messages=messages,
            max_tokens=MAX_TOKENS_QUESTION,
            temperature=TEMPERATURE_INTERVIEW,
         )
         return {
            "response":          text.strip(),
            "tokens_used":       tokens,
            "interview_done":    False,
            "rag_used":          False,
            "method":            "prebuilt",
            "next_question_id":  next_q["id"],
            "all_questions_done": False,
            "followup_asked":    False,
         }

    # Normal path: candidate just answered the last bank question.
    # Classify their answer → route to praise+followup OR ack+next.
    is_strong = await _classify_answer(
        question=last_q["question"] if last_q else "",
        answer=user_text,
    )

    has_followup = bool(last_q and last_q.get("follow_up"))

    if is_strong and has_followup:
        # STRONG answer + follow-up available → praise specifically + probe
        log.info(f"[PREBUILT] STRONG answer to {last_qid} → praise + follow-up")
        system_content = PRAISE_AND_FOLLOWUP_PROMPT.format(
            follow_up=last_q["follow_up"],
            answer=user_text[:400],
        )
        messages = [{"role": "system", "content": system_content}] + history[-4:]
        text, tokens = await llm_chat(
            messages=messages,
            max_tokens=MAX_TOKENS_QUESTION,
            temperature=TEMPERATURE_INTERVIEW,
        )
        return {
            "response":          text.strip(),
            "tokens_used":       tokens,
            "interview_done":    False,
            "rag_used":          False,
            "method":            "prebuilt",
            "next_question_id":  last_qid,   # still on same bank question
            "all_questions_done": False,
            "followup_asked":    True,        # mark: next answer moves us on
        }

    # WEAK answer (or no follow-up available) → ack + move to next bank question
    log.info(f"[PREBUILT] {'WEAK' if not is_strong else 'NO_FOLLOWUP'} for {last_qid} → next question")
    next_q = next((q for q in questions if q["id"] not in asked_question_ids), None)
    if next_q is None:
        return _exhausted_response()

    system_content = ACKNOWLEDGE_AND_NEXT_PROMPT.format(next_question=next_q["question"])
    messages = [{"role": "system", "content": system_content}] + history[-4:]
    text, tokens = await llm_chat(
        messages=messages,
        max_tokens=MAX_TOKENS_QUESTION,
        temperature=TEMPERATURE_INTERVIEW,
    )
    return {
        "response":          text.strip(),
        "tokens_used":       tokens,
        "interview_done":    False,
        "rag_used":          False,
        "method":            "prebuilt",
        "next_question_id":  next_q["id"],
        "all_questions_done": False,
        "followup_asked":    False,
    }


def _exhausted_response() -> Dict:
    """Bank just got exhausted — invite Q&A phase."""
    return {
        "response": (
            "That's all the questions I had for you. Thanks for the thoughtful "
            "answers. Now, do you have any questions for me about the role or "
            "the team?"
        ),
        "tokens_used":       0,
        "interview_done":    False,
        "rag_used":          False,
        "method":            "prebuilt",
        "next_question_id":  None,
        "all_questions_done": True,
        "followup_asked":    False,
    }