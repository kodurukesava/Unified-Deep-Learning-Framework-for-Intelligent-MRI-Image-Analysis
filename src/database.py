import sqlite3
from datetime import datetime

from config import DATABASE_PATH


def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            gradcam_path TEXT,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()

    connection.close()


def save_prediction(
    filename,
    prediction,
    confidence,
    gradcam_path
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (
            filename,
            prediction,
            confidence,
            gradcam_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        filename,
        prediction,
        confidence,
        gradcam_path,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ))

    connection.commit()

    connection.close()


def get_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows