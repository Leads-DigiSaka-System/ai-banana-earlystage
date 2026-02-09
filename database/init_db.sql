-- Tables for feedback enhancement (run after create_database.sql).
-- Run: psql -U ai_banana_user -d ai_banana_early_stage -f database/init_db.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Predictions (every inference saved)
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    user_location VARCHAR(255),
    image_path VARCHAR(500) NOT NULL,
    image_size_kb INTEGER,
    image_width INTEGER,
    image_height INTEGER,
    image_hash VARCHAR(64),
    predicted_class_id INTEGER NOT NULL,
    predicted_class_name VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_data JSONB,
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    inference_time_ms FLOAT,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_predictions_user_timestamp ON predictions(user_id, timestamp);
CREATE INDEX idx_predictions_class ON predictions(predicted_class_name);
CREATE INDEX idx_predictions_model ON predictions(model_version);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp DESC);

-- User feedback (correct/wrong + correct class)
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    is_correct BOOLEAN NOT NULL,
    correct_class_id INTEGER,
    correct_class_name VARCHAR(50),
    user_comment TEXT,
    confidence_rating INTEGER CHECK (confidence_rating BETWEEN 1 AND 5),
    feedback_source VARCHAR(50),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    processed_for_training BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP
);

CREATE INDEX idx_feedback_prediction ON feedback(prediction_id);
CREATE INDEX idx_feedback_correct ON feedback(is_correct);
CREATE INDEX idx_feedback_processed ON feedback(processed_for_training);
CREATE INDEX idx_feedback_timestamp ON feedback(timestamp DESC);

-- Training data (from feedback / app for retraining)
CREATE TABLE training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_path VARCHAR(500) NOT NULL,
    image_hash VARCHAR(64) UNIQUE,
    class_id INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    bbox_data JSONB,
    source VARCHAR(50) NOT NULL,
    source_id UUID,
    quality_score FLOAT,
    blur_score FLOAT,
    brightness_score FLOAT,
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by VARCHAR(255),
    validated_at TIMESTAMP,
    dataset_split VARCHAR(20),
    added_date TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_training_class ON training_data(class_name);
CREATE INDEX idx_training_source ON training_data(source);
CREATE INDEX idx_training_validated ON training_data(is_validated);
CREATE INDEX idx_training_split ON training_data(dataset_split);
CREATE INDEX idx_training_date ON training_data(added_date DESC);

-- Model performance from feedback (aggregated)
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_predictions INTEGER NOT NULL,
    total_feedback INTEGER NOT NULL,
    correct_predictions INTEGER NOT NULL,
    incorrect_predictions INTEGER NOT NULL,
    accuracy FLOAT,
    class_metrics JSONB,
    avg_confidence FLOAT,
    avg_confidence_correct FLOAT,
    avg_confidence_incorrect FLOAT,
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_performance_model_date ON model_performance(model_version, date);
CREATE INDEX idx_performance_date ON model_performance(date DESC);
