
# ============================================================
# UNIFIED DEEP LEARNING FRAMEWORK FOR INTELLIGENT MRI ANALYSIS
# Flask Application
# ============================================================

import os
import sqlite3
import json
import logging

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
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DATABASE_DIR, "mri_analysis.db")

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv(
    os.path.join(BASE_DIR, ".env"),
    override=False
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not SECRET_KEY:
    SECRET_KEY = os.urandom(32).hex()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

app.secret_key = SECRET_KEY


# ============================================================
# GROQ AI CONFIGURATION
# ============================================================

groq_client = None

if GROQ_API_KEY:

    try:

        groq_client = Groq(
            api_key=GROQ_API_KEY
        )

        logger.info("Groq AI initialized successfully.")

    except Exception:

        logger.exception("Groq initialization failed.")

else:

    logger.warning("GROQ_API_KEY is not configured.")


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:

    init_dashboard_db()

    logger.info("Dashboard database initialized.")

except Exception:

    logger.exception("Database initialization failed.")


# ============================================================
# MRI MODEL INITIALIZATION
# ============================================================

predictor = None

try:

    predictor = MRIPredictor()

    logger.info("MRI model loaded successfully.")

    logger.info(
        "MRI classes: %s",
        getattr(predictor, "class_names", [])
    )

except Exception:

    logger.exception("MRI model loading failed.")


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


SUPPORTED_LANGUAGES = {
    "English",
    "Telugu",
    "Hindi",
    "Tamil",
    "Kannada",
    "Malayalam",
    "Bengali",
    "Marathi",
    "Gujarati",
    "Punjabi",
    "Urdu"
}


LANGUAGE_INSTRUCTIONS = {

    "English": "Answer in simple English.",

    "Telugu": "Answer in natural Telugu.",

    "Hindi": "Answer in natural Hindi.",

    "Tamil": "Answer in natural Tamil.",

    "Kannada": "Answer in natural Kannada.",

    "Malayalam": "Answer in natural Malayalam.",

    "Bengali": "Answer in natural Bengali.",

    "Marathi": "Answer in natural Marathi.",

    "Gujarati": "Answer in natural Gujarati.",

    "Punjabi": "Answer in natural Punjabi.",

    "Urdu": "Answer in natural Urdu."
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


def render_result_error(message, status_code=400):

    return render_template(
        "result.html",
        error=message,
        image=None,
        prediction=None,
        confidence=0,
        probabilities={},
        analysis_id=None
    ), status_code


def normalize_dashboard_data(data):
    """
    Normalize dashboard data returned by dashboard_db.py.

    Supports:
    - Dictionary
    - Tuple containing a dictionary
    - Tuple containing stats and additional data
    """

    if isinstance(data, dict):

        dashboard_data = data.copy()

    elif isinstance(data, tuple):

        dictionaries = [
            item for item in data
            if isinstance(item, dict)
        ]

        if not dictionaries:

            raise TypeError(
                "get_dashboard_data() returned a tuple "
                "without a dictionary."
            )

        dashboard_data = {}

        for item in dictionaries:
            dashboard_data.update(item)

    else:

        raise TypeError(
            "Unexpected dashboard data type: "
            + type(data).__name__
        )

    stats = dashboard_data.get("stats", {})

    if not isinstance(stats, dict):

        stats = {}

    # Support dashboard functions that return statistics
    # directly rather than inside a "stats" dictionary.

    if not stats:

        possible_keys = (
            "patients",
            "total_patients",
            "scans",
            "total_scans",
            "predictions",
            "total_predictions"
        )

        stats = {
            key: dashboard_data[key]
            for key in possible_keys
            if key in dashboard_data
        }

    dashboard_data["stats"] = stats

    return dashboard_data


def get_safe_dashboard_data():

    try:

        raw_data = get_dashboard_data()

        return normalize_dashboard_data(raw_data)

    except Exception:

        logger.exception(
            "Could not retrieve dashboard data."
        )

        raise


# ============================================================
# ROLE SELECTION
# ============================================================

@app.route("/roles")
def role_selection():

    return render_template("role_selection.html")


@app.route("/open-scanner/<role>")
def open_scanner(role):

    allowed_roles = {
        "doctor",
        "admin",
        "patient"
    }

    if role not in allowed_roles:

        return redirect(
            url_for("role_selection")
        )

    session["selected_role"] = role

    return redirect(
        url_for("index")
    )


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

    patient_name = request.form.get(
        "patient_name", ""
    ).strip()

    patient_code = request.form.get(
        "patient_code", ""
    ).strip()

    patient_age_text = request.form.get(
        "patient_age", ""
    ).strip()

    if not all([
        patient_name,
        patient_code,
        patient_age_text
    ]):

        return render_result_error(
            "Please enter all patient details.",
            400
        )

    if len(patient_name) > 100 or len(patient_code) > 50:

        return render_result_error(
            "Patient details are too long.",
            400
        )

    try:

        patient_age = int(patient_age_text)

        if not 1 <= patient_age <= 120:
            raise ValueError

    except (ValueError, TypeError):

        return render_result_error(
            "Enter a valid patient age.",
            400
        )

    if predictor is None:

        return render_result_error(
            "MRI model is not available.",
            503
        )

    if "mri_image" not in request.files:

        return render_result_error(
            "No MRI image was uploaded.",
            400
        )

    uploaded_file = request.files["mri_image"]

    if not uploaded_file.filename:

        return render_result_error(
            "Please select an MRI image.",
            400
        )

    if not allowed_file(uploaded_file.filename):

        return render_result_error(
            "Unsupported image format.",
            400
        )

    safe_filename = secure_filename(
        uploaded_file.filename
    )

    if not safe_filename:

        return render_result_error(
            "Invalid image filename.",
            400
        )

    stem, extension = os.path.splitext(
        safe_filename
    )

    filename = (
        f"{stem}_{os.urandom(6).hex()}"
        f"{extension.lower()}"
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        uploaded_file.save(image_path)

        result = predictor.predict(image_path)

        if not isinstance(result, dict):

            raise TypeError(
                "Invalid prediction result."
            )

        prediction_label = result.get(
            "prediction",
            "Unknown"
        )

        probabilities = result.get(
            "probabilities",
            {}
        )

        if not isinstance(probabilities, dict):

            probabilities = {}

        try:

            confidence = round(
                float(result.get("confidence", 0)),
                2
            )

        except (ValueError, TypeError):

            confidence = 0

        # Save patient and analysis details.

        connection = sqlite3.connect(DB_PATH)

        connection.row_factory = sqlite3.Row

        try:

            now = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            patient = connection.execute(
                """
                SELECT id
                FROM patients
                WHERE patient_code = ?
                """,
                (patient_code,)
            ).fetchone()

            if patient:

                patient_id = patient["id"]

                connection.execute(
                    """
                    UPDATE patients
                    SET name = ?, age = ?
                    WHERE id = ?
                    """,
                    (
                        patient_name,
                        patient_age,
                        patient_id
                    )
                )

            else:

                cursor = connection.execute(
                    """
                    INSERT INTO patients
                    (
                        patient_code,
                        name,
                        age,
                        created_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        patient_code,
                        patient_name,
                        patient_age,
                        now
                    )
                )

                patient_id = cursor.lastrowid

            cursor = connection.execute(
                """
                INSERT INTO analyses
                (
                    patient_id,
                    image_filename,
                    prediction,
                    confidence,
                    probabilities,
                    analysis_date
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    filename,
                    str(prediction_label),
                    confidence,
                    json.dumps(probabilities),
                    now
                )
            )

            analysis_id = cursor.lastrowid

            connection.commit()

        except Exception:

            connection.rollback()

            raise

        finally:

            connection.close()

        logger.info(
            "MRI analysis saved. Patient: %s, Analysis: %s",
            patient_code,
            analysis_id
        )

        return render_template(
            "result.html",
            image=filename,
            prediction=prediction_label,
            confidence=confidence,
            probabilities=probabilities,
            analysis_id=analysis_id,
            error=None,
            selected_role=session.get(
                "selected_role",
                "patient"
            )
        )

    except Exception:

        logger.exception(
            "MRI prediction or database error."
        )

        return render_result_error(
            "MRI analysis failed. Please check server logs.",
            500
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
            error=None
        )

    prediction_label = request.form.get(
        "prediction",
        "Unknown"
    ).strip()

    confidence = request.form.get(
        "confidence",
        "0"
    ).strip()

    language = request.form.get(
        "language",
        "English"
    ).strip()

    question = request.form.get(
        "question",
        ""
    ).strip()

    if language not in SUPPORTED_LANGUAGES:

        language = "English"

    template_data = {
        "prediction": prediction_label,
        "confidence": confidence,
        "language": language,
        "question": question,
        "response": None,
        "error": None
    }

    if groq_client is None:

        template_data["error"] = (
            "Groq AI is not configured. "
            "Check the GROQ_API_KEY environment variable."
        )

        return render_template(
            "chat.html",
            **template_data
        )

    if not question:

        template_data["error"] = (
            "Please enter a question."
        )

        return render_template(
            "chat.html",
            **template_data
        )

    language_instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["English"]
    )

    system_prompt = f"""
You are an educational MRI AI assistant.

Application prediction: {prediction_label}
Model confidence: {confidence}%

LANGUAGE:
{language_instruction}

Respond in the selected language.

MEDICAL SAFETY:

1. AI predictions are not confirmed diagnoses.
2. Do not prescribe medicines or dosages.
3. Do not advise starting or stopping medication.
4. Explain MRI concepts in simple language.
5. Recommend evaluation by a qualified radiologist.
6. For emergency symptoms, recommend urgent medical care.
7. Never create unnecessary fear.
"""

    try:

        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.2,
            max_completion_tokens=1200
        )

        response = ""

        if completion.choices:

            response = (
                completion.choices[0].message.content
                or ""
            )

        template_data["response"] = (
            response.strip()
            or "Please try asking your question again."
        )

    except Exception:

        logger.exception(
            "Groq chatbot error."
        )

        template_data["error"] = (
            "The AI assistant could not generate a response."
        )

    return render_template(
        "chat.html",
        **template_data
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    try:

        dashboard_data = get_safe_dashboard_data()

        stats = dashboard_data.get(
            "stats",
            {}
        )

        return render_template(
            "dashboard.html",
            dashboard_data=dashboard_data,
            stats=stats
        )

    except Exception:

        logger.exception(
            "Dashboard loading failed."
        )

        flash(
            "Could not load dashboard data.",
            "error"
        )

        return render_template(
            "dashboard.html",
            dashboard_data={},
            stats={
                "patients": 0,
                "scans": 0,
                "predictions": 0,
                "total_patients": 0,
                "total_scans": 0,
                "total_predictions": 0
            }
        ), 500


# ============================================================
# PATIENTS LIST
# ============================================================

@app.route("/patients")
def patients():

    if not os.path.isfile(DB_PATH):

        flash(
            "Patient database was not found.",
            "error"
        )

        return render_template(
            "patients.html",
            patients=[]
        ), 500

    connection = None

    try:

        connection = sqlite3.connect(DB_PATH)

        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
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
        ).fetchall()

        patient_list = [
            dict(row)
            for row in rows
        ]

        return render_template(
            "patients.html",
            patients=patient_list
        )

    except Exception:

        logger.exception(
            "Patient list loading failed."
        )

        flash(
            "Could not load patient records.",
            "error"
        )

        return render_template(
            "patients.html",
            patients=[]
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

        if isinstance(patient, tuple):

            dictionaries = [
                item for item in patient
                if isinstance(item, dict)
            ]

            if dictionaries:
                patient = dictionaries[0]

        if isinstance(patient, sqlite3.Row):

            patient = dict(patient)

        if not patient:

            abort(404)

        return render_template(
            "patient.html",
            patient=patient
        )

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "Patient details loading failed."
        )

        abort(500)


# ============================================================
# ANALYSIS DETAILS
# ============================================================

@app.route("/analysis/<int:analysis_id>")
def analysis_details(analysis_id):

    try:

        analysis = get_analysis(analysis_id)

        if isinstance(analysis, tuple):

            dictionaries = [
                item for item in analysis
                if isinstance(item, dict)
            ]

            if dictionaries:
                analysis = dictionaries[0]

        if isinstance(analysis, sqlite3.Row):

            analysis = dict(analysis)

        if not analysis:

            abort(404)

        return render_template(
            "analysis.html",
            analysis=analysis
        )

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "Analysis details loading failed."
        )

        abort(500)


# ============================================================
# PDF REPORT DOWNLOAD
# ============================================================

@app.route("/report/<int:analysis_id>")
def download_report(analysis_id):

    try:

        analysis = get_analysis(analysis_id)

        if not analysis:

            return "Analysis not found.", 404

        pdf_output = generate_pdf_report(
            analysis
        )

        if pdf_output is None:

            return "PDF report could not be generated.", 500

        if hasattr(pdf_output, "read"):

            pdf_output.seek(0)

            return send_file(
                pdf_output,
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"MRI_Report_{analysis_id}.pdf"
            )

        if isinstance(
            pdf_output,
            (str, bytes, os.PathLike)
        ):

            pdf_path = os.fsdecode(
                pdf_output
            )

            if not os.path.isabs(pdf_path):

                pdf_path = os.path.join(
                    BASE_DIR,
                    pdf_path
                )

            if not os.path.isfile(pdf_path):

                return "PDF file not found.", 500

            return send_file(
                pdf_path,
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"MRI_Report_{analysis_id}.pdf"
            )

        return "Unsupported PDF report format.", 500

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "PDF report generation failed."
        )

        return "PDF report generation failed.", 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    classes = []

    if predictor is not None:

        classes = getattr(
            predictor,
            "class_names",
            []
        )

    model_status = (
        "loaded"
        if predictor is not None
        else "not loaded"
    )

    chatbot_status = (
        "configured"
        if groq_client is not None
        else "not configured"
    )

    return {
        "status": "running",
        "model": model_status,
        "classes": classes,
        "chatbot": chatbot_status,
        "groq_model": GROQ_MODEL
    }, 200


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_result_error(
        "MRI image exceeds the 16 MB upload limit.",
        413
    )


@app.errorhandler(404)
def page_not_found(error):

    return render_result_error(
        "Requested page was not found.",
        404
    )


@app.errorhandler(500)
def internal_server_error(error):

    return render_result_error(
        "An internal server error occurred. Please check the server logs.",
        500
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "UNIFIED DEEP LEARNING FRAMEWORK "
        "FOR INTELLIGENT MRI IMAGE ANALYSIS"
    )

    print("=" * 65)

    print(
        "MRI model:",
        "Loaded" if predictor is not None else "Not loaded"
    )

    print(
        "Groq chatbot:",
        "Ready" if groq_client is not None else "Not configured"
    )

    print("Groq model:", GROQ_MODEL)

    print("Database:", DB_PATH)

    print("=" * 65)

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
