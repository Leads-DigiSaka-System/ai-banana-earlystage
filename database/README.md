# Database setup (PostgreSQL) — ai_banana_early_stage

## Option A: Python script (recommended)

1. In `.env` set:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=ai_banana_early_stage
POSTGRES_USER=postgres
POSTGRES_PASSWORD=012345
```

2. From project root:

```bash
pip install -r requirements.txt
python -m scripts.setup_database
```

First run creates the database (if missing) then creates all tables. If tables already exist, re-run may error; DB is already ready.

## Option B: psql manual

### 1. Create database and user (run once)

From project root, as Postgres superuser:

```bash
psql -U postgres -f database/create_database.sql
```

Edit `database/create_database.sql` to set the password for `ai_banana_user`, or change it after:

```bash
psql -U postgres -d ai_banana_early_stage -c "ALTER USER ai_banana_user WITH PASSWORD 'your_password';"
```

### 2. Create tables

```bash
psql -U ai_banana_user -d ai_banana_early_stage -f database/init_db.sql
```

## 3. .env

Use the same `POSTGRES_*` vars; the app can build `DATABASE_URL` from them when the DB layer is wired.

## Tables

| Table              | Purpose                          |
|--------------------|----------------------------------|
| predictions        | Each inference (image, class, confidence) |
| feedback           | User says correct/wrong + correct class   |
| training_data      | Images/labels for retraining             |
| model_performance  | Aggregated accuracy by model/date        |

---

## Data dictionary

Source of truth: `database/init_db.sql` and `database/models.py`.

### predictions

| Column | Type | Purpose |
|--------|------|---------|
| id | UUID | Primary key. |
| user_id | VARCHAR(255) | Who made the request (required). |
| user_location | VARCHAR(255) | Optional (e.g. city). |
| image_path | VARCHAR(500) | Path to stored image (local or MinIO). |
| image_size_kb, image_width, image_height | INT | Optional image metadata. |
| image_hash | VARCHAR(64) | Optional dedup/checksum. |
| predicted_class_id | INT | Class index (0–6). |
| predicted_class_name | VARCHAR(50) | e.g. Stage2, Healthy. |
| confidence | FLOAT | Model confidence. |
| bbox_data | JSONB | Bounding boxes (if full detection). |
| model_version | VARCHAR(50) | e.g. v1.0. |
| model_name | VARCHAR(100) | e.g. YOLO12n. |
| inference_time_ms | FLOAT | Optional. |
| timestamp | TIMESTAMP | When saved. |
| metadata | JSONB | Optional extra. |

**Relationships:** One prediction can have many `feedback` rows (feedback.prediction_id → predictions.id).

### feedback

| Column | Type | Purpose |
|--------|------|---------|
| id | UUID | Primary key. |
| prediction_id | UUID | FK → predictions(id), ON DELETE CASCADE. |
| is_correct | BOOLEAN | User said correct/wrong. |
| correct_class_id | INT | If wrong, correct class index. |
| correct_class_name | VARCHAR(50) | If wrong, e.g. Stage3. |
| user_comment | TEXT | Optional. |
| confidence_rating | INT | Optional 1–5 (CHECK 1–5). |
| feedback_source | VARCHAR(50) | Optional. |
| processed_for_training | BOOLEAN | Whether used for training_data. |
| processed_at | TIMESTAMP | When processed. |
| timestamp | TIMESTAMP | When submitted. |

### training_data

| Column | Type | Purpose |
|--------|------|---------|
| id | UUID | Primary key. |
| image_path | VARCHAR(500) | Same path as prediction (local or MinIO). |
| image_hash | VARCHAR(64) | UNIQUE, optional dedup. |
| class_id | INT | Class index (0–6). |
| class_name | VARCHAR(50) | e.g. Stage2. |
| bbox_data | JSONB | Bbox for YOLO labels. |
| source | VARCHAR(50) | e.g. 'feedback'. Export uses source='feedback'. |
| source_id | UUID | Optional link to prediction/feedback. |
| quality_score, blur_score, brightness_score | FLOAT | Optional. |
| is_validated | BOOLEAN | Optional. |
| dataset_split | VARCHAR(20) | e.g. train, val. |
| added_date | TIMESTAMP | When added. |
| metadata | JSONB | Optional. |

### model_performance

| Column | Type | Purpose |
|--------|------|---------|
| id | UUID | Primary key. |
| model_version | VARCHAR(50) | e.g. v1.0. |
| model_name | VARCHAR(100) | e.g. YOLO12n. |
| date | DATE | Aggregation date. |
| total_predictions, total_feedback | INT | Counts. |
| correct_predictions, incorrect_predictions | INT | From feedback. |
| accuracy | FLOAT | correct / total_feedback. |
| class_metrics | JSONB | Per-class stats. |
| avg_confidence* | FLOAT | Optional averages. |
| calculated_at | TIMESTAMP | When computed. |

UNIQUE on (model_version, date).
