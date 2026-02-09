# Create database tables (predictions, feedback, training_data).
# Run: python -m scripts.init_db
# TODO: Use database.init_db() when DB is configured.

if __name__ == "__main__":
    from database import init_db
    init_db()
    print("Database init done.")
