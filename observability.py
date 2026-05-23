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

#  MLFLOW SETUP
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"[MLFLOW] Tracking to: {MLFLOW_TRACKING_URI}")

#  LANGSMITH SETUP
if LANGSMITH_API_KEY:
    print(f"[LANGSMITH] Tracing enabled — view at https://smith.langchain.com")
else:
    print(f"[LANGSMITH] Disabled (set LANGSMITH_API_KEY to enable)")

#  TURN TRACKING — log every interview turn to MLflow
@contextmanager
def track_turn(session_id: str, turn_number: int):

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

#  SESSION TRACKING — log the parent session
def start_session_run(session_id: str, email: str):
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