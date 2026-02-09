# ModelPerformance table (separate module to avoid SQLAlchemy scan issue in tests).

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from database.models import Base


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    total_feedback: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    incorrect_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    class_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_confidence_correct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_confidence_incorrect: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calculated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="NOW()")
