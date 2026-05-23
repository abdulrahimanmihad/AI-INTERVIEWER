
import json
import uuid
from typing import Optional, Tuple

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Text, Integer, DateTime, select, func

from config import DATABASE_URL

#  ENGINE — connection pool
# Convert sync URL to async URL for SQLAlchemy 2.0 async
# sqlite:///   → sqlite+aiosqlite:///
# postgresql:// → postgresql+asyncpg://
if DATABASE_URL.startswith("sqlite:"):
    ASYNC_URL = DATABASE_URL.replace("sqlite:", "sqlite+aiosqlite:", 1)
    IS_SQLITE = True
elif DATABASE_URL.startswith("postgresql:"):
    ASYNC_URL = DATABASE_URL.replace("postgresql:", "postgresql+asyncpg:", 1)
    IS_SQLITE = False
else:
    ASYNC_URL = DATABASE_URL
    IS_SQLITE = False

if IS_SQLITE:
    engine = create_async_engine(
        ASYNC_URL,
        echo=False,    # set True for SQL debug output
    )
else:
    engine = create_async_engine(
        ASYNC_URL,
        pool_size=10,
        max_overflow=5,
        pool_pre_ping=True,
        echo=False,
    )

# Session factory — gives us AsyncSession instances
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()
print(f"[DB] Using {'SQLite' if IS_SQLITE else 'PostgreSQL'}")


#  MODELS — Python classes that map to DB tables
class User(Base):
    """Registered candidates."""
    __tablename__ = "users"

    session_id = Column(String, primary_key=True)
    full_name  = Column(String)
    email      = Column(String, unique=True, index=True)


class CompletedInterview(Base):
    """Finished interviews with full transcript."""
    __tablename__ = "completed_interviews"

    session_id      = Column(String, primary_key=True)
    final_summary   = Column(Text)
    full_transcript = Column(Text)    # stored as JSON string
    tokens_total    = Column(Integer, default=0)
    rag_method      = Column(String)  # which method was used (for analytics)
    llm_provider    = Column(String)
    completed_at    = Column(DateTime, server_default=func.now())


#  INIT
async def init_db():
    """Create tables if they don't exist. Called once at startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[DB] Tables initialized")


#  OPERATIONS
async def check_interview_status(email: str) -> Tuple[str, Optional[str]]:
    """
    Check if email completed an interview.

    Returns:
        ("ALREADY_COMPLETED", None) | ("ELIGIBLE", session_id) | ("ELIGIBLE", None)
    """
    async with async_session() as session:
        # Check for completed interview by email
        result = await session.execute(
            select(CompletedInterview.session_id, User.email)
            .join(User, User.session_id == CompletedInterview.session_id)
            .where(User.email == email)
        )
        if result.first():
            return ("ALREADY_COMPLETED", None)

        # Check for registered user
        result = await session.execute(
            select(User.session_id).where(User.email == email)
        )
        row = result.first()
        return ("ELIGIBLE", row[0] if row else None)


async def register_new_user(full_name: str, email: str) -> str:
    """Register a new candidate, return new session_id."""
    new_id = str(uuid.uuid4())
    async with async_session() as session:
        session.add(User(session_id=new_id, full_name=full_name, email=email))
        try:
            await session.commit()
        except Exception:
            # Email already exists (race condition) — fetch existing
            await session.rollback()
            result = await session.execute(
                select(User.session_id).where(User.email == email)
            )
            row = result.first()
            return row[0] if row else new_id
    return new_id


async def archive_interview(
    session_id: str,
    state: dict,
    rag_method: str,
    llm_provider: str,
):
    """
    Archive completed interview.

    CRITICAL: caller deletes Redis AFTER this returns successfully.
    """
    async with async_session() as session:
        record = CompletedInterview(
            session_id      = session_id,
            final_summary   = state.get("summary", ""),
            full_transcript = json.dumps(state.get("history", [])),
            tokens_total    = state.get("tokens_total", 0),
            rag_method      = rag_method,
            llm_provider    = llm_provider,
        )
        await session.merge(record)
        await session.commit()
        print(f"[DB] Archived {session_id}")