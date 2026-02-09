-- Run once as superuser (e.g. postgres) to create DB and user.
-- Example: psql -U postgres -f database/create_database.sql

-- Replace password with your own, or set via: ALTER USER ai_banana_user WITH PASSWORD 'yours';
CREATE USER ai_banana_user WITH PASSWORD 'change_me_in_env';

CREATE DATABASE ai_banana_early_stage
    OWNER ai_banana_user
    ENCODING 'UTF8';

GRANT ALL PRIVILEGES ON DATABASE ai_banana_early_stage TO ai_banana_user;

-- Allow create in public schema (Postgres 15+)
\c ai_banana_early_stage
GRANT ALL ON SCHEMA public TO ai_banana_user;
GRANT CREATE ON SCHEMA public TO ai_banana_user;
