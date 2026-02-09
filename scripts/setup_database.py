"""
Create database and tables (predictions, feedback, etc.).
Run from project root: python -m scripts.setup_database

Requires .env with:
  POSTGRES_HOST=localhost
  POSTGRES_PORT=5433
  POSTGRES_DB=ai_banana_early_stage
  POSTGRES_USER=postgres
  POSTGRES_PASSWORD=your_password
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def make_url(host: str, port: str, db: str, user: str, password: str) -> str:
    from urllib.parse import quote_plus
    pw = quote_plus(password)
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def get_conn(database_url: str):
    import psycopg2
    return psycopg2.connect(database_url)


def run_sql_file(conn, filepath: Path):
    sql = filepath.read_text(encoding="utf-8")
    lines = []
    for line in sql.splitlines():
        s = line.strip()
        if s.startswith("--") or s.startswith("\\"):
            continue
        lines.append(line)
    block = "\n".join(lines)
    statements = [s.strip() for s in block.split(";") if s.strip()]
    cur = conn.cursor()
    for stmt in statements:
        if stmt:
            cur.execute(stmt)
    cur.close()
    conn.commit()


def main():
    load_env()

    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB")
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")

    if not all([db, user, password]):
        print("ERROR: .env must have POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
        sys.exit(1)
    if not re.match(r"^[a-zA-Z0-9_]+$", db) or not re.match(r"^[a-zA-Z0-9_]+$", user):
        print("ERROR: POSTGRES_DB and POSTGRES_USER must be alphanumeric + underscore only.")
        sys.exit(1)

    # Step 1: Create database if not exists (connect to default DB)
    print("Step 1: Ensuring database exists...")
    url_default = make_url(host, port, "postgres", user, password)
    try:
        conn_admin = get_conn(url_default)
        conn_admin.autocommit = True
        cur = conn_admin.cursor()
        try:
            cur.execute(f"""
                CREATE DATABASE {db}
                OWNER {user}
                ENCODING 'UTF8';
            """)
            print(f"  Created database {db}.")
        except Exception as e:
            if "already exists" in str(e):
                print(f"  Database {db} already exists.")
            else:
                raise
        cur.close()
        conn_admin.close()
    except Exception as e:
        print(f"  Step 1 failed: {e}")
        sys.exit(1)

    # Step 2: Create tables
    print("Step 2: Creating tables...")
    init_sql = ROOT / "database" / "init_db.sql"
    if not init_sql.exists():
        print(f"  ERROR: {init_sql} not found.")
        sys.exit(1)
    url_app = make_url(host, port, db, user, password)
    try:
        conn = get_conn(url_app)
        run_sql_file(conn, init_sql)
        conn.close()
        print("  Tables created: predictions, feedback, training_data, model_performance.")
    except Exception as e:
        print(f"  Step 2 failed: {e}")
        sys.exit(1)

    print(f"Done. Database {db} is ready.")


if __name__ == "__main__":
    main()
