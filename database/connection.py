# Database connection and session. Feedback enhancement uses PostgreSQL.

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Build URL from POSTGRES_* env (same as scripts/setup_database.py)
def _get_database_url() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "ai_banana_early_stage")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    from urllib.parse import quote_plus
    pw = quote_plus(password)
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


try:
    from dotenv import load_dotenv
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL") or _get_database_url()

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=_get_engine()
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield DB session."""
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables from SQLAlchemy models (optional; we use init_db.sql)."""
    from database.models import Base
    import database.model_performance  # noqa: F401 - register ModelPerformance
    Base.metadata.create_all(bind=_get_engine())
