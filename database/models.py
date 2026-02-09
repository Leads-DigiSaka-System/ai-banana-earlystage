# SQLAlchemy models for feedback enhancement (PostgreSQL). Match database/init_db.sql.

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    image_size_kb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    predicted_class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_class_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("NOW()"), index=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    feedbacks: Mapped[list["Feedback"]] = relationship(
        "Feedback", back_populates="prediction", cascade="all, delete-orphan"
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    correct_class_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    correct_class_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    feedback_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("NOW()"), index=True
    )
    processed_for_training: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            text("confidence_rating IS NULL OR (confidence_rating >= 1 AND confidence_rating <= 5)"),
            name="feedback_confidence_rating_check",
        ),
    )

    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="feedbacks")


class TrainingData(Base):
    __tablename__ = "training_data"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    image_hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    bbox_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    blur_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brightness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    validated_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    validated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    dataset_split: Mapped[Optional[str]] = mapped_column(String(20), index=True, nullable=True)
    added_date: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=text("NOW()"), index=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)
