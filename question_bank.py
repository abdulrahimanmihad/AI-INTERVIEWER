"""
question_bank.py

Loads per-company interview question banks from CSV files and indexes them
into a dedicated ChromaDB collection (one collection per company).

CSV schema (header row required):
    question,category,role,follow_up

  question  : the interview question text (required)
  category  : free-form tag like "background", "technical", "behavioral"
  role      : e.g. "ml_engineer", "backend", "*"  (use "*" for any role)
  follow_up : optional canned follow-up if candidate answer is vague

CSV files live in ./question_banks/<company_id>.csv

The module exposes:
    load_company_bank(company_id) -> bool   # True if loaded
    get_all_questions(company_id) -> list[dict]   # fixed order
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)

BANKS_DIR = Path("./question_banks")
CHROMA_DIR = "./chroma_question_banks"

# one chroma client for all company collections
_chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

# in-memory cache: company_id -> ordered list of question dicts
_bank_cache: dict = {}


def _collection_name(company_id: str) -> str:
    # chroma collection names: 3-63 chars, alnum/dash/underscore
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in company_id)
    return f"qbank_{safe}"[:63]


def load_company_bank(company_id: str, role_filter: Optional[str] = None) -> bool:
    """
    Read ./question_banks/<company_id>.csv, validate schema, cache the
    ordered question list in memory, and index into a per-company ChromaDB
    collection. Returns True if at least one question was loaded.

    role_filter: if provided, only load rows where role == role_filter or "*"
    """
    csv_path = BANKS_DIR / f"{company_id}.csv"
    if not csv_path.exists():
        log.warning(f"[QBANK] No bank file for {company_id} at {csv_path}")
        return False

    questions: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question", "category", "role", "follow_up"}
        if not required.issubset(set(reader.fieldnames or [])):
            log.error(f"[QBANK] {csv_path} missing required columns {required}")
            return False
        for i, row in enumerate(reader):
            q = (row.get("question") or "").strip()
            if not q:
                continue
            role = (row.get("role") or "*").strip().lower()
            if role_filter and role not in (role_filter.lower(), "*"):
                continue
            questions.append({
                "id":        f"{company_id}_q{i:03d}",
                "question":  q,
                "category":  (row.get("category") or "general").strip(),
                "role":      role,
                "follow_up": (row.get("follow_up") or "").strip(),
                "order":     i,
            })

    if not questions:
        log.warning(f"[QBANK] No usable rows in {csv_path}")
        return False

    # cache in-memory for fixed-order retrieval
    _bank_cache[company_id] = questions

    # index into chroma (useful for future similarity-based selection)
    try:
        coll = _chroma_client.get_or_create_collection(name=_collection_name(company_id))
        # wipe and rewrite for idempotency on re-load
        existing = coll.get()
        if existing.get("ids"):
            coll.delete(ids=existing["ids"])
        coll.add(
            ids=[q["id"] for q in questions],
            documents=[q["question"] for q in questions],
            metadatas=[{
                "category": q["category"],
                "role":     q["role"],
                "order":    q["order"],
                "follow_up": q["follow_up"],
            } for q in questions],
        )
    except Exception as e:
        log.error(f"[QBANK] Chroma index failed for {company_id}: {e}")
        # cache is still usable for fixed-order, so we don't fail hard

    log.info(f"[QBANK] Loaded {len(questions)} questions for company '{company_id}'")
    return True


def get_all_questions(company_id: str) -> List[dict]:
    """Return the ordered question list (loads from CSV if not yet cached)."""
    if company_id not in _bank_cache:
        load_company_bank(company_id)
    return _bank_cache.get(company_id, [])


def question_count(company_id: str) -> int:
    return len(get_all_questions(company_id))