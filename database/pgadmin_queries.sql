-- ============================================
-- Banana Feedback DB - Useful Queries
-- ============================================
-- Database: ai_banana_early_stage
-- Copy and paste these in pgAdmin Query Tool
-- ============================================

-- 1. List all tables in public schema
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_type = 'BASE TABLE'
ORDER BY table_name;

-- 2. SELECT - Predictions (all, latest first)
SELECT
    id,
    user_id,
    user_location,
    image_path,
    predicted_class_id,
    predicted_class_name,
    confidence,
    model_version,
    model_name,
    inference_time_ms,
    timestamp
FROM predictions
ORDER BY timestamp DESC;

-- 3. SELECT - Predictions with search (by user_id)
SELECT
    id,
    user_id,
    predicted_class_name,
    confidence,
    timestamp
FROM predictions
WHERE user_id ILIKE '%your_user%'   -- Replace your_user
ORDER BY timestamp DESC;

-- 4. SELECT - Predictions with search (by class)
SELECT
    id,
    user_id,
    predicted_class_name,
    confidence,
    timestamp
FROM predictions
WHERE predicted_class_name ILIKE '%Stage2%'   -- Replace Stage1, Stage2, Healthy, etc.
ORDER BY timestamp DESC;

-- 5. SELECT - Predictions with search (by model_version)
SELECT
    id,
    user_id,
    predicted_class_name,
    confidence,
    model_version,
    timestamp
FROM predictions
WHERE model_version = 'v1.0'   -- Replace with your version
ORDER BY timestamp DESC;

-- 6. SELECT - One prediction by ID
SELECT *
FROM predictions
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 7. UPDATE - Prediction (e.g. fix user_location or class)
UPDATE predictions
SET user_location = 'Manila',
    predicted_class_name = 'Stage3',
    predicted_class_id = 3,
    confidence = 0.92
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 8. DELETE - One prediction (cascade deletes related feedback)
DELETE FROM predictions
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 9. SELECT - Feedback (all, latest first)
SELECT
    id,
    prediction_id,
    is_correct,
    correct_class_id,
    correct_class_name,
    user_comment,
    confidence_rating,
    feedback_source,
    processed_for_training,
    timestamp
FROM feedback
ORDER BY timestamp DESC;

-- 10. SELECT - Feedback for one prediction
SELECT
    id,
    prediction_id,
    is_correct,
    correct_class_name,
    user_comment,
    timestamp
FROM feedback
WHERE prediction_id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'   -- Replace with prediction UUID
ORDER BY timestamp DESC;

-- 11. SELECT - Feedback where user said wrong
SELECT
    f.id,
    f.prediction_id,
    f.is_correct,
    f.correct_class_name,
    p.predicted_class_name,
    p.user_id,
    f.timestamp
FROM feedback f
JOIN predictions p ON p.id = f.prediction_id
WHERE f.is_correct = FALSE
ORDER BY f.timestamp DESC;

-- 12. UPDATE - Feedback (e.g. mark as processed for training)
UPDATE feedback
SET processed_for_training = TRUE,
    processed_at = NOW()
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 13. DELETE - One feedback row
DELETE FROM feedback
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 14. SELECT - Training data (all)
SELECT
    id,
    image_path,
    image_hash,
    class_id,
    class_name,
    source,
    dataset_split,
    is_validated,
    added_date
FROM training_data
ORDER BY added_date DESC;

-- 15. SELECT - Training data with search (by class_name)
SELECT
    id,
    image_path,
    class_name,
    source,
    dataset_split,
    added_date
FROM training_data
WHERE class_name ILIKE '%Stage%'   -- Replace with Healthy, Stage1, etc.
ORDER BY added_date DESC;

-- 16. SELECT - Training data by source (e.g. from feedback)
SELECT
    id,
    image_path,
    class_name,
    source,
    added_date
FROM training_data
WHERE source = 'feedback'   -- or 'original', 'app', 'manual'
ORDER BY added_date DESC;

-- 17. UPDATE - Training data (e.g. set split or validated)
UPDATE training_data
SET dataset_split = 'train',
    is_validated = TRUE,
    validated_by = 'admin',
    validated_at = NOW()
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 18. DELETE - One training_data row
DELETE FROM training_data
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 19. SELECT - Model performance (all)
SELECT
    id,
    model_version,
    model_name,
    date,
    total_predictions,
    total_feedback,
    correct_predictions,
    incorrect_predictions,
    accuracy,
    avg_confidence,
    calculated_at
FROM model_performance
ORDER BY date DESC;

-- 20. SELECT - Model performance with search (by version / date range)
SELECT
    id,
    model_version,
    date,
    total_feedback,
    correct_predictions,
    incorrect_predictions,
    accuracy,
    calculated_at
FROM model_performance
WHERE model_version = 'v1.0'   -- Replace
  AND date >= '2025-01-01'
  AND date <= '2025-12-31'
ORDER BY date DESC;

-- 21. UPDATE - Model performance (e.g. fix accuracy)
UPDATE model_performance
SET accuracy = 0.85,
    avg_confidence = 0.82
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 22. DELETE - One model_performance row
DELETE FROM model_performance
WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx';   -- Replace with real UUID

-- 23. Get statistics (counts)
SELECT
    (SELECT COUNT(*) FROM predictions)       AS total_predictions,
    (SELECT COUNT(*) FROM feedback)          AS total_feedback,
    (SELECT COUNT(*) FROM training_data)     AS total_training_data,
    (SELECT COUNT(*) FROM model_performance) AS total_model_performance;

-- 24. Predictions per class
SELECT
    predicted_class_name,
    COUNT(*) AS count,
    ROUND(AVG(confidence)::numeric, 4) AS avg_confidence
FROM predictions
GROUP BY predicted_class_name
ORDER BY count DESC;

-- 25. Feedback accuracy (correct vs incorrect)
SELECT
    is_correct,
    COUNT(*) AS count
FROM feedback
GROUP BY is_correct;

-- 26. Check table structure - predictions
SELECT
    column_name,
    data_type,
    character_maximum_length,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'predictions'
ORDER BY ordinal_position;

-- 27. Check table structure - feedback
SELECT
    column_name,
    data_type,
    character_maximum_length,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'feedback'
ORDER BY ordinal_position;

-- 28. Check table structure - training_data
SELECT
    column_name,
    data_type,
    character_maximum_length,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'training_data'
ORDER BY ordinal_position;

-- 29. Check table structure - model_performance
SELECT
    column_name,
    data_type,
    character_maximum_length,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'model_performance'
ORDER BY ordinal_position;
