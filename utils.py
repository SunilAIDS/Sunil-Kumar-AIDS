import sqlite3
import bcrypt
from pathlib import Path

DB_FILE = "database.db"
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# -------------------- Database Setup --------------------
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
    c.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
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
    conn.commit()
    conn.close()

# -------------------- User Functions --------------------
def create_user(username, password, role):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    if c.fetchone():
        conn.close()
        return False
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed, role))
    conn.commit()
    conn.close()
    return True

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode(), user[2]):
        return {"user_id": user[0], "username": user[1], "role": user[3]}
    return None

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users")
    users = c.fetchall()
    conn.close()
    return users

# -------------------- Upload Functions --------------------
def save_upload(user_id, img_path, gradcam_path, prediction):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO uploads (user_id, img_path, gradcam_path, prediction) VALUES (?, ?, ?, ?)",
        (user_id, img_path, gradcam_path, prediction)
    )
    conn.commit()
    conn.close()

def get_user_uploads(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE user_id=? ORDER BY upload_id DESC", (user_id,))
    uploads = c.fetchall()
    conn.close()
    return uploads

def get_all_uploads():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM uploads ORDER BY upload_id DESC")
    uploads = c.fetchall()
    conn.close()
    return uploads

def verify_upload(upload_id, comments):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE uploads SET verified=1, comments=? WHERE upload_id=?", (comments, upload_id))
    conn.commit()
    conn.close()

# -------------------- Initialize DB --------------------
create_tables()
