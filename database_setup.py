import sqlite3

DB_FILE = "database.db"

def create_tables():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    # Uploads table
    # Check if 'uploads' table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='uploads'")
    if not c.fetchone():
        # Table does not exist, create new with all columns
        c.execute("""
        CREATE TABLE uploads (
            upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            img_path TEXT,
            gradcam_path TEXT,
            prediction TEXT,
            verified INTEGER DEFAULT 0,
            comments TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """)
    else:
        # Table exists, check if 'comments' column exists
        c.execute("PRAGMA table_info(uploads)")
        columns = [col[1] for col in c.fetchall()]
        if "comments" not in columns:
            c.execute("ALTER TABLE uploads ADD COLUMN comments TEXT")

    conn.commit()
    conn.close()
    print("Database setup completed successfully!")

if __name__ == "__main__":
    create_tables()
