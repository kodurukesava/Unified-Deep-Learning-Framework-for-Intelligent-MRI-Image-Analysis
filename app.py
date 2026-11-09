# ============================================================
# MRI HEALTH CHATBOT + MRI IMAGE ANALYSIS
# ============================================================

import os
import sqlite3
import json
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    abort,
    flash,
    session,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv
from groq import Groq

from src.predict import MRIPredictor
from config import UPLOAD_DIR

from src.dashboard_db import (
    init_dashboard_db,
    get_dashboard_data,
    get_patient,
    get_analysis,
)

from src.report_generator import generate_pdf_report


# ============================================================
# PATHS AND ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DATABASE_DIR, "mri_analysis.db")

load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = None

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq API client initialized.")
        print("Groq model:", GROQ_MODEL)
    except Exception as error:
        print("Groq initialization error:", error)
else:
    print("WARNING: GROQ_API_KEY was not found in .env")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "change-this-development-key",
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:
    os.makedirs(DATABASE_DIR, exist_ok=True)
    init_dashboard_db()
    print("Dashboard database initialized.")
except Exception as error:
    print("Database initialization error:", error)


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "bmp", "webp"
}

SUPPORTED_LANGUAGES = {
    "English", "Telugu", "Hindi", "Tamil", "Kannada",
    "Malayalam", "Bengali", "Marathi", "Gujarati",
    "Punjabi", "Urdu",
}

LANGUAGE_INSTRUCTIONS = {
    "English": "Answer completely in clear, simple English.",
    "Telugu": "Answer in natural, simple Telugu (తెలుగు). Use English only for necessary medical terms.",
    "Hindi": "Answer in natural, simple Hindi (हिन्दी). Use English only for necessary medical terms.",
    "Tamil": "Answer in natural, simple Tamil (தமிழ்). Use English only for necessary medical terms.",
    "Kannada": "Answer in natural, simple Kannada (ಕನ್ನಡ). Use English only for necessary medical terms.",
    "Malayalam": "Answer in natural, simple Malayalam (മലയാളം). Use English only for necessary medical terms.",
    "Bengali": "Answer in natural, simple Bengali (বাংলা). Use English only for necessary medical terms.",
    "Marathi": "Answer in natural, simple Marathi (मराठी). Use English only for necessary medical terms.",
    "Gujarati": "Answer in natural, simple Gujarati (ગુજરાતી). Use English only for necessary medical terms.",
    "Punjabi": "Answer in natural, simple Punjabi (ਪੰਜਾਬੀ). Use English only for necessary medical terms.",
    "Urdu": "Answer in natural, simple Urdu (اردو). Use English only for necessary medical terms.",
}


# ============================================================
# LOAD MRI MODEL
# ============================================================

print("=" * 65)
print("INTELLIGENT MRI IMAGE ANALYSIS SYSTEM")
print("=" * 65)

predictor = None

try:
    predictor = MRIPredictor()
    print("MRI model loaded successfully.")

    try:
        print("Classes:", predictor.class_names)
    except Exception:
        pass

except Exception as error:
    print("MRI model loading error:", error)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):
    """Check whether the uploaded image has an allowed extension."""

    if not filename or "." not in filename:
        return False

    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Integrated error-rendering function supplied by you.
def render_result_error(message, status_code=400):
    return render_template(
        "result.html",
        error=message,
        image=None,
        prediction=None,
        confidence=0,
        probabilities={},
        analysis_id=None,
    ), status_code


# ============================================================
# ROLE SELECTION
# ============================================================

@app.route("/roles")
def role_selection():
    return render_template("role_selection.html")


@app.route("/open-scanner/<role>")
def open_scanner(role):
    allowed_roles = {"doctor", "admin", "patient"}

    if role not in allowed_roles:
        return redirect(url_for("role_selection"))

    session["selected_role"] = role

    return redirect(url_for("index"))


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# MRI IMAGE PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    # Validate patient details submitted by templates/index.html.
    patient_name = request.form.get("patient_name", "").strip()
    patient_code = request.form.get("patient_code", "").strip()
    patient_age_text = request.form.get("patient_age", "").strip()

    if not patient_name or not patient_code or not patient_age_text:
        return render_result_error(
            "Please enter patient name, patient code, and age.", 400
        )

    if len(patient_name) > 100 or len(patient_code) > 50:
        return render_result_error(
            "Patient name or patient code is too long.", 400
        )

    try:
        patient_age = int(patient_age_text)
        if not 1 <= patient_age <= 120:
            raise ValueError("Age must be between 1 and 120.")
    except (ValueError, TypeError):
        return render_result_error("Please enter a valid patient age.", 400)

    if predictor is None:
        return render_result_error(
            "MRI model could not be loaded. Check the model file and terminal.",
            503,
        )

    if "mri_image" not in request.files:
        return render_result_error("No MRI image was uploaded.", 400)

    uploaded_file = request.files["mri_image"]

    if not uploaded_file.filename:
        return render_result_error("Please select an MRI image.", 400)

    if not allowed_file(uploaded_file.filename):
        return render_result_error(
            "Invalid format. Upload JPG, JPEG, PNG, BMP or WEBP.", 400
        )

    safe_filename = secure_filename(uploaded_file.filename)
    if not safe_filename:
        return render_result_error("Invalid filename.", 400)

    # Make the filename unique to avoid overwriting an earlier scan.
    stem, extension = os.path.splitext(safe_filename)
    filename = f"{stem}_{os.urandom(6).hex()}{extension.lower()}"

    try:
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(image_path)
    except OSError as error:
        print("Image upload error:", error)
        return render_result_error(
            "Could not save the uploaded MRI image.", 500
        )

    try:
        result = predictor.predict(image_path)

        if not isinstance(result, dict):
            return render_result_error(
                "The MRI model returned an invalid result.", 500
            )

        prediction_label = result.get("prediction", "Unknown")
        probabilities = result.get("probabilities", {})

        try:
            confidence = round(float(result.get("confidence", 0)), 2)
        except (ValueError, TypeError):
            confidence = 0

        if not isinstance(probabilities, dict):
            probabilities = {}

        # Save patient and analysis records after prediction succeeds.
        # The column names must match src/dashboard_db.py's database schema.
        os.makedirs(DATABASE_DIR, exist_ok=True)
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row

        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            patient = connection.execute(
                "SELECT id FROM patients WHERE patient_code = ?",
                (patient_code,),
            ).fetchone()

            if patient:
                patient_id = patient["id"]
                connection.execute(
                    """
                    UPDATE patients
                    SET name = ?, age = ?
                    WHERE id = ?
                    """,
                    (patient_name, patient_age, patient_id),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO patients
                        (patient_code, name, age, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (patient_code, patient_name, patient_age, now),
                )
                patient_id = cursor.lastrowid

            cursor = connection.execute(
                """
                INSERT INTO analyses
                    (patient_id, image_filename, prediction,
                     confidence, probabilities, analysis_date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    filename,
                    str(prediction_label),
                    confidence,
                    json.dumps(probabilities),
                    now,
                ),
            )
            analysis_id = cursor.lastrowid
            connection.commit()

        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        print("=" * 65)
        print("MRI PREDICTION RESULT")
        print("Patient code:", patient_code)
        print("Prediction:", prediction_label)
        print("Confidence:", confidence, "%")
        print("Analysis ID:", analysis_id)
        print("=" * 65)

        return render_template(
            "result.html",
            image=filename,
            prediction=prediction_label,
            confidence=confidence,
            probabilities=probabilities,
            analysis_id=analysis_id,
            error=None,
            selected_role=session.get("selected_role", "patient"),
        )

    except Exception as error:
        print(
            "MRI prediction/database error:",
            type(error).__name__,
            str(error),
        )
        return render_result_error(
            "An error occurred while analyzing the MRI image or saving its record. "
            "Check the terminal for details.",
            500,
        )


# ============================================================
# MULTILINGUAL MRI AI CHATBOT
# ============================================================

@app.route("/chat", methods=["GET", "POST"])
def chat():

    if request.method == "GET":
        return render_template(
            "chat.html",
            prediction="Unknown",
            confidence=0,
            language="English",
            question="",
            response=None,
            error=None,
        )

    prediction_label = request.form.get(
        "prediction", "Unknown"
    ).strip()

    confidence = request.form.get(
        "confidence", "0"
    ).strip()

    language = request.form.get(
        "language", "English"
    ).strip()

    question = request.form.get(
        "question", ""
    ).strip()

    if language not in SUPPORTED_LANGUAGES:
        language = "English"

    template_data = {
        "prediction": prediction_label,
        "confidence": confidence,
        "language": language,
        "question": question,
        "response": None,
        "error": None,
    }

    if groq_client is None:
        template_data["error"] = (
            "Groq AI is not configured. "
            "Check GROQ_API_KEY in your .env file."
        )

        return render_template("chat.html", **template_data)

    if not question:
        template_data["error"] = "Please enter a question."

        return render_template("chat.html", **template_data)

    language_instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["English"],
    )

    system_prompt = f"""
You are an educational MRI AI assistant.

Application model prediction: {prediction_label}
Model confidence: {confidence}%
Selected response language: {language}

LANGUAGE REQUIREMENT:
{language_instruction}

Respond in the selected language even if the question
is written in another language. Use simple language.

MEDICAL SAFETY:
- An AI classification is not a confirmed diagnosis.
- Do not claim the patient definitely has or does not
  have a disease based only on the prediction.
- Do not prescribe medicines or dosages.
- Do not advise starting or stopping medication.
- Give educational information, not a replacement for
  advice from a qualified healthcare professional.
- Recommend professional medical evaluation.
- For severe or emergency symptoms, recommend urgent care.
- Avoid unsupported claims and unnecessary alarm.
- Answer the question directly and concisely.
"""

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )

        response = ""

        if completion.choices:
            response = (
                completion.choices[0].message.content or ""
            )

        template_data["response"] = (
            response.strip()
            or "The AI assistant did not return a response. Please try again."
        )

        return render_template("chat.html", **template_data)

    except Exception as error:
        print(
            "Groq chatbot error:",
            type(error).__name__,
            str(error),
        )

        template_data["error"] = (
            "The AI assistant could not generate a response. "
            "Check the terminal for details."
        )

        return render_template("chat.html", **template_data)


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    try:
        dashboard_data = get_dashboard_data()

        return render_template(
            "dashboard.html",
            dashboard_data=dashboard_data,
            stats=dashboard_data.get("stats", {}),
        )

    except Exception as error:
        print(
            "Dashboard error:",
            type(error).__name__,
            str(error),
        )

        flash(
            "Could not load dashboard data. Check the terminal and database.",
            "error",
        )

        return render_template(
            "dashboard.html",
            dashboard_data={},
            stats={
                "patients": 0,
                "scans": 0,
                "predictions": 0,
            },
        ), 500


# ============================================================
# PATIENTS LIST
# ============================================================

@app.route("/patients")
def patients():
    """Display patients and their latest MRI analysis."""

    if not os.path.isfile(DB_PATH):
        flash(
            "Patient database was not found. Check your database configuration.",
            "error",
        )

        return render_template(
            "patients.html",
            patients=[],
        ), 500

    connection = None

    try:
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row

        sql = """
            SELECT
                p.id AS patient_id,
                p.patient_code,
                p.name,
                p.age,
                p.created_at,
                COUNT(a.id) AS total_scans,
                MAX(a.analysis_date) AS last_analysis
            FROM patients AS p
            LEFT JOIN analyses AS a
                ON a.patient_id = p.id
            GROUP BY
                p.id,
                p.patient_code,
                p.name,
                p.age,
                p.created_at
            ORDER BY p.id DESC
        """

        rows = connection.execute(sql).fetchall()

        patient_list = [
            dict(row) for row in rows
        ]

        return render_template(
            "patients.html",
            patients=patient_list,
        )

    except sqlite3.Error as error:
        print(
            "Patients database error:",
            type(error).__name__,
            str(error),
        )

        flash(
            "Could not load patients. Check the database schema.",
            "error",
        )

        return render_template(
            "patients.html",
            patients=[],
        ), 500

    finally:
        if connection is not None:
            connection.close()


# ============================================================
# PATIENT DETAILS
# ============================================================

@app.route("/patient/<int:patient_id>")
def patient_details(patient_id):

    try:
        patient = get_patient(patient_id)

        if not patient:
            abort(404)

        return render_template(
            "patient.html",
            patient=patient,
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Patient lookup error:",
            type(error).__name__,
            str(error),
        )

        abort(500)


# ============================================================
# ANALYSIS DETAILS
# ============================================================

@app.route("/analysis/<int:analysis_id>")
def analysis_details(analysis_id):

    try:
        analysis = get_analysis(analysis_id)

        if not analysis:
            abort(404)

        return render_template(
            "analysis.html",
            analysis=analysis,
        )

    except HTTPException:
        raise

    except Exception as error:
        print(
            "Analysis lookup error:",
            type(error).__name__,
            str(error),
        )

        abort(500)


# ============================================================
# PDF REPORT DOWNLOAD
# ============================================================

@app.route("/report/<int:analysis_id>")
def download_report(analysis_id):
    """Generate and download a PDF report for a saved MRI analysis.

    Supports either an in-memory file object (such as BytesIO) or a PDF
    file path returned by generate_pdf_report().
    """
    try:
        analysis = get_analysis(analysis_id)

        if not analysis:
            return "Analysis not found.", 404

        # Pass the analysis record to the report generator.
        pdf_output = generate_pdf_report(analysis)

        if pdf_output is None:
            app.logger.error(
                "PDF generator returned no output for analysis %s",
                analysis_id,
            )
            return "PDF report could not be generated.", 500

        # Handle an in-memory PDF (for example, io.BytesIO).
        if hasattr(pdf_output, "seek") and hasattr(pdf_output, "read"):
            pdf_output.seek(0)
            return send_file(
                pdf_output,
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"MRI_Report_{analysis_id}.pdf",
            )

        # Handle a PDF saved to a file path.
        if isinstance(pdf_output, (str, bytes, os.PathLike)):
            pdf_path = os.fsdecode(pdf_output)

            if not os.path.isabs(pdf_path):
                pdf_path = os.path.join(BASE_DIR, pdf_path)

            if not os.path.isfile(pdf_path):
                app.logger.error(
                    "Generated PDF file was not found: %s",
                    pdf_path,
                )
                return "PDF file was not found.", 500

            return send_file(
                pdf_path,
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"MRI_Report_{analysis_id}.pdf",
            )

        app.logger.error(
            "Unsupported PDF output type for analysis %s: %s",
            analysis_id,
            type(pdf_output).__name__,
        )
        return "Unsupported PDF report format.", 500

    except HTTPException:
        raise

    except Exception:
        app.logger.exception(
            "PDF report download failed for analysis %s",
            analysis_id,
        )
        return "PDF download failed. Check the Flask terminal.", 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    classes = []

    if predictor is not None:
        try:
            classes = predictor.class_names
        except Exception:
            pass

    status_code = 200 if predictor is not None else 503

    return {
        "status": "running",
        "model": (
            "loaded"
            if predictor is not None
            else "not loaded"
        ),
        "classes": classes,
        "chatbot": (
            "configured"
            if groq_client is not None
            else "not configured"
        ),
        "groq_model": GROQ_MODEL,
    }, status_code


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    return render_result_error(
        "The uploaded MRI image is too large. Maximum size is 16 MB.",
        413,
    )


@app.errorhandler(404)
def page_not_found(error):
    return render_result_error(
        "The requested page was not found.",
        404,
    )


@app.errorhandler(500)
def internal_server_error(error):
    return render_result_error(
        "An internal server error occurred. Please check the terminal.",
        500,
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("Starting Flask MRI Analysis System...")
    print(
        "MRI model:",
        "Loaded" if predictor is not None else "Not loaded",
    )
    print(
        "Groq chatbot:",
        "Ready" if groq_client is not None else "Not configured",
    )
    print("Groq model:", GROQ_MODEL)
    print("Open http://127.0.0.1:5000")
    print("=" * 65)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
