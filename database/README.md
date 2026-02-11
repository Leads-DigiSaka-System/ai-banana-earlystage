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
