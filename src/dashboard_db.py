
# src/dashboard_db.py

import json
import os
import sqlite3
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "mri_analysis.db")


def get_connection():
    os.makedirs(DATABASE_DIR, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_dashboard_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                age INTEGER,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                image_filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT,
                analysis_date TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)


def save_analysis(
    patient_code,
    patient_name,
    age,
    image_filename,
    prediction,
    confidence,
    probabilities
):
    patient_code = patient_code.strip()
    patient_name = patient_name.strip()

    if not patient_code or not patient_name:
        raise ValueError("Patient ID and name are required.")

    if age is not None and not 0 <= int(age) <= 120:
        raise ValueError("Age must be between 0 and 120.")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO patients (patient_code, name, age, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(patient_code) DO UPDATE SET
                name = excluded.name,
                age = excluded.age
            """,
            (
                patient_code,
                patient_name,
                int(age) if age not in (None, "") else None,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

        patient = conn.execute(
            "SELECT id FROM patients WHERE patient_code = ?",
            (patient_code,)
        ).fetchone()

        cursor = conn.execute(
            """
            INSERT INTO analyses (
                patient_id, image_filename, prediction,
                confidence, probabilities, analysis_date
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patient["id"],
                image_filename,
                str(prediction),
                float(confidence),
                json.dumps(probabilities or {}),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )

        return cursor.lastrowid


def get_dashboard_data():
    with get_connection() as conn:
        stats = {
            "patients": conn.execute(
                "SELECT COUNT(*) FROM patients"
            ).fetchone()[0],
            "scans": conn.execute(
                "SELECT COUNT(*) FROM analyses"
            ).fetchone()[0],
            "today": conn.execute(
                """
                SELECT COUNT(*) FROM analyses
                WHERE date(analysis_date) = date('now', 'localtime')
                """
            ).fetchone()[0]
        }

        recent = conn.execute("""
            SELECT
                a.id AS analysis_id,
                p.patient_code,
                p.name AS patient_name,
                a.prediction,
                a.confidence,
                a.analysis_date
            FROM analyses a
            JOIN patients p ON p.id = a.patient_id
            ORDER BY a.id DESC
            LIMIT 100
        """).fetchall()

        return stats, [dict(row) for row in recent]


def get_patient(patient_id):
    with get_connection() as conn:
        patient = conn.execute(
            "SELECT * FROM patients WHERE id = ?",
            (patient_id,)
        ).fetchone()

        if patient is None:
            return None, []

        analyses = conn.execute("""
            SELECT * FROM analyses
            WHERE patient_id = ?
            ORDER BY id DESC
        """, (patient_id,)).fetchall()

        return dict(patient), [dict(row) for row in analyses]


def get_analysis(analysis_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                a.*,
                p.patient_code,
                p.name AS patient_name,
                p.age AS patient_age
            FROM analyses a
            JOIN patients p ON p.id = a.patient_id
            WHERE a.id = ?
        """, (analysis_id,)).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["probabilities"] = json.loads(
            result.get("probabilities") or "{}"
        )
        return result
