"""
TSM-RAG Data Models — SQLAlchemy ORM for log storage.

Phase 1: Simple schema for raw log storage in SQLite.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Text, create_engine, Index
)
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()


class Log(Base):
    """Raw test log entry."""

    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(32), unique=True, nullable=True)  # External ID (e.g., LOG-20260521-001)
    stand_id = Column(String(16), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    test_id = Column(String(32), nullable=True)
    result = Column(String(8), nullable=False)  # PASS, FAIL, ERROR
    error_code = Column(Integer, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    voltage = Column(Float, nullable=True)
    current = Column(Float, nullable=True)
    message = Column(Text, nullable=True)
    raw_data = Column(Text, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_stand", "stand_id"),
        Index("idx_result", "result"),
        Index("idx_timestamp", "timestamp"),
        Index("idx_error_code", "error_code"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.log_id or f"LOG-{self.id:06d}",
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "stand_id": self.stand_id,
            "test_id": self.test_id,
            "result": self.result,
            "error_code": self.error_code,
            "message": self.message,
            "voltage": self.voltage,
            "current": self.current,
            "duration_ms": self.duration_ms,
        }


class Event(Base):
    """Triggered event (when a rule matched)."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(Integer, nullable=True)  # FK to logs.id
    stand_id = Column(String(16), nullable=False)
    rule_name = Column(String(64), nullable=False)
    severity = Column(String(16), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Integer, default=0)  # 0 = pending, 1 = acked
    message = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_event_stand", "stand_id"),
        Index("idx_event_severity", "severity"),
        Index("idx_event_timestamp", "timestamp"),
    )


def init_db(db_path: str = "data/logs.db"):
    """Initialize SQLite database and create tables."""
    import os

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get a new SQLAlchemy session."""
    return Session(engine)
