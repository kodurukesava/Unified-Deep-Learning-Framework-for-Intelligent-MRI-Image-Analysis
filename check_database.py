
import os
import sqlite3

db_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "database",
    "mri_analysis.db"
)

connection = sqlite3.connect(db_path)
connection.row_factory = sqlite3.Row

print("\n--- PATIENT RECORDS ---")
patients = connection.execute(
    "SELECT * FROM patients"
).fetchall()

print("Total patients:", len(patients))

for patient in patients:
    print(dict(patient))

print("\n--- MRI ANALYSES ---")
analyses = connection.execute(
    "SELECT * FROM analyses"
).fetchall()

print("Total analyses:", len(analyses))

for analysis in analyses:
    print(dict(analysis))

connection.close()
