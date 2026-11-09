
import os
import sqlite3

db_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "database",
    "mri_analysis.db"
)

connection = sqlite3.connect(db_path)

try:
    for table in ["patients", "analyses"]:
        count = connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count} records")

finally:
    connection.close()
