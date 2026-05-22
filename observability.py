"""
observability.py
═════════════════════════════════════════════════════════════════════
MLflow + LangSmith — tracking experiments and LLM calls.

WHY THIS MATTERS FOR HIM:
    He explicitly asked to compare RAG methods.
    "Which works better?" needs DATA, not opinion.
    MLflow logs every turn: which method, how many tokens, how fast.
    LangSmith shows every LLM call in detail.
    After running 20 interviews per method, you have real numbers to compare.

WHAT EACH TOOL DOES:

    MLflow:
        - Open source, self-hosted (or use Databricks)
        - Free locally — data saved to ./mlruns folder
        - Best for: comparing experiments side-by-side
        - Run `mlflow ui` and open http://localhost:5000 to see results

    LangSmith:
        - SaaS from the LangChain team
        - Free tier: 5000 traces/month
        - Best for: debugging individual LLM calls
        - Auto-traces every LangChain/LangGraph call when API key is set

WHAT IF YOU DIDN'T USE THESE:
    You're guessing which RAG method works best
    No way to prove latency improvements with data
    Debugging "why did the LLM say X?" is impossible without traces
═════════════════════════════════════════════════════════════════════
"""

import time
from contextlib import contextmanager

import mlflow

from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    LANGSMITH_API_KEY,
    RAG_METHOD,
    LLM_PROVIDER,
)


# ═════════════════════════════════════════════════════════════════
#  MLFLOW SETUP
# ═════════════════════════════════════════════════════════════════
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"[MLFLOW] Tracking to: {MLFLOW_TRACKING_URI}")


# ═════════════════════════════════════════════════════════════════
#  LANGSMITH SETUP
# ═════════════════════════════════════════════════════════════════
# LangSmith is enabled if API key is set (see config.py — it auto-sets env vars)
# When enabled, LangChain/LangGraph traces auto-send to LangSmith
# We don't need to do anything else here — just verify it's on
if LANGSMITH_API_KEY:
    print(f"[LANGSMITH] Tracing enabled — view at https://smith.langchain.com")
else:
    print(f"[LANGSMITH] Disabled (set LANGSMITH_API_KEY to enable)")


# ═════════════════════════════════════════════════════════════════
#  TURN TRACKING — log every interview turn to MLflow
# ═════════════════════════════════════════════════════════════════
@contextmanager
def track_turn(session_id: str, turn_number: int):
    """
    Context manager that logs one interview turn to MLflow.

    USAGE:
        with track_turn(session_id, turn_num) as log:
            result = await run_turn(...)
            log.metrics(
                tokens=result["tokens_used"],
                rag_used=result["rag_used"],
            )

    WHAT GETS LOGGED:
        - Which RAG method was used
        - Which LLM provider was used
        - How long the turn took (latency)
        - How many tokens were used
        - Whether RAG retrieval happened

    WHY CONTEXT MANAGER:
        Auto-times the turn (start/end markers)
        Auto-handles errors (logs them as failed runs)
    """

    class Logger:
        def metrics(self, **kwargs):
            for k, v in kwargs.items():
                # MLflow only accepts numbers as metrics
                if isinstance(v, bool):
                    v = 1 if v else 0
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
                else:
                    mlflow.log_param(k, str(v))

    log = Logger()
    run_name = f"{session_id[:8]}_turn_{turn_number}"

    with mlflow.start_run(run_name=run_name, nested=True):
        # Tag the run so you can filter by method/provider in the UI
        mlflow.log_param("rag_method",   RAG_METHOD)
        mlflow.log_param("llm_provider", LLM_PROVIDER)
        mlflow.log_param("session_id",   session_id)
        mlflow.log_param("turn_number",  turn_number)

        start = time.time()
        try:
            yield log
            mlflow.log_metric("latency_seconds", time.time() - start)
            mlflow.log_metric("success", 1)
        except Exception as e:
            mlflow.log_metric("latency_seconds", time.time() - start)
            mlflow.log_metric("success", 0)
            mlflow.log_param("error", str(e)[:200])
            raise


# ═════════════════════════════════════════════════════════════════
#  SESSION TRACKING — log the parent session
# ═════════════════════════════════════════════════════════════════
def start_session_run(session_id: str, email: str):
    """
    Start a parent MLflow run for the whole interview.
    Each turn becomes a nested run.

    BUG FIX:
        If the previous WebSocket connection died without ending the run
        (which happens on browser close), MLflow still thinks the run is
        active. Calling start_run() again raises "Run is already active".
        Fix: end any active run BEFORE starting a new one — safe regardless
        of whether one exists.

    Returns the run object so the caller can end it later.
    """
    # Defensive: kill any leftover active run from a crashed previous session
    # mlflow.active_run() returns None if no run is active — safe to call always
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass  # any error here shouldn't block starting the new run

    run = mlflow.start_run(run_name=f"interview_{session_id[:8]}")
    mlflow.log_param("session_id",   session_id)
    mlflow.log_param("email",        email)
    mlflow.log_param("rag_method",   RAG_METHOD)
    mlflow.log_param("llm_provider", LLM_PROVIDER)
    return run


def end_session_run(total_turns: int, total_tokens: int, completed: bool):
    """Log final session metrics and end the parent run."""
    try:
        mlflow.log_metric("total_turns",   total_turns)
        mlflow.log_metric("total_tokens",  total_tokens)
        mlflow.log_metric("completed",     1 if completed else 0)
        mlflow.end_run()
    except Exception:
        # If somehow already ended, don't crash the app
        pass