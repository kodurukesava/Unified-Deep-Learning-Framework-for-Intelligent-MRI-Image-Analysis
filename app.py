# ============================================================
# MRI HEALTH CHATBOT + MRI IMAGE ANALYSIS
# ============================================================

import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

from src.predict import MRIPredictor
from config import UPLOAD_DIR


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = None

if GROQ_API_KEY:

    try:
        groq_client = Groq(
            api_key=GROQ_API_KEY
        )

        print("=" * 70)
        print("Groq API key loaded successfully.")
        print("Groq model:", GROQ_MODEL)
        print("=" * 70)

    except Exception as error:

        print("=" * 70)
        print("ERROR: Could not initialize Groq client.")
        print("Error:", error)
        print("=" * 70)

else:

    print("=" * 70)
    print("WARNING: GROQ_API_KEY was not found.")
    print("Please check your .env file.")
    print("=" * 70)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# ALLOWED IMAGE FORMATS
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


# ============================================================
# SUPPORTED LANGUAGES
# ============================================================

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


# ============================================================
# LANGUAGE INSTRUCTIONS
# ============================================================

LANGUAGE_INSTRUCTIONS = {

    "English": (
        "Answer completely in English. "
        "Use clear and simple English. "
        "Do not switch to another language."
    ),

    "Telugu": (
        "Answer completely in Telugu (తెలుగు). "
        "Use natural, simple and easy-to-understand Telugu. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Hindi": (
        "Answer completely in Hindi (हिन्दी). "
        "Use natural, simple and easy-to-understand Hindi. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Tamil": (
        "Answer completely in Tamil (தமிழ்). "
        "Use natural, simple and easy-to-understand Tamil. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Kannada": (
        "Answer completely in Kannada (ಕನ್ನಡ). "
        "Use natural, simple and easy-to-understand Kannada. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Malayalam": (
        "Answer completely in Malayalam (മലയാളം). "
        "Use natural, simple and easy-to-understand Malayalam. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Bengali": (
        "Answer completely in Bengali (বাংলা). "
        "Use natural, simple and easy-to-understand Bengali. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Marathi": (
        "Answer completely in Marathi (मराठी). "
        "Use natural, simple and easy-to-understand Marathi. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Gujarati": (
        "Answer completely in Gujarati (ગુજરાતી). "
        "Use natural, simple and easy-to-understand Gujarati. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Punjabi": (
        "Answer completely in Punjabi (ਪੰਜਾਬੀ). "
        "Use natural, simple and easy-to-understand Punjabi. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    ),

    "Urdu": (
        "Answer completely in Urdu (اردو). "
        "Use natural, simple and easy-to-understand Urdu. "
        "Do not translate the answer into English. "
        "Use English only for necessary medical terms."
    )
}


# ============================================================
# STARTUP
# ============================================================

print()
print("=" * 70)
print("        INTELLIGENT MRI IMAGE ANALYSIS SYSTEM")
print("=" * 70)


# ============================================================
# LOAD MRI MODEL
# ============================================================

print()
print("Loading trained MRI model...")

try:

    predictor = MRIPredictor()

    print("MRI model loaded successfully.")

    try:
        print(
            "Classes:",
            predictor.class_names
        )
    except Exception:
        pass

except Exception as error:

    predictor = None

    print()
    print("ERROR: Could not load MRI model.")
    print("Error:", error)


# ============================================================
# FILE EXTENSION CHECK
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# MRI PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if predictor is None:

        return render_template(
            "result.html",
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error=(
                "MRI model could not be loaded. "
                "Please check the model file and terminal."
            )
        )


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "mri_image" not in request.files:

        return render_template(
            "result.html",
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error="No MRI image was uploaded."
        )


    file = request.files["mri_image"]


    # --------------------------------------------------------
    # CHECK FILE NAME
    # --------------------------------------------------------

    if not file.filename:

        return render_template(
            "result.html",
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error="Please select an MRI image."
        )


    # --------------------------------------------------------
    # CHECK FORMAT
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        return render_template(
            "result.html",
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error=(
                "Invalid file format. "
                "Please upload JPG, JPEG, PNG, BMP or WEBP."
            )
        )


    # --------------------------------------------------------
    # SAFE FILE NAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    if not filename:

        return render_template(
            "result.html",
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error="Invalid filename."
        )


    # --------------------------------------------------------
    # CREATE UPLOAD DIRECTORY
    # --------------------------------------------------------

    try:

        os.makedirs(
            app.config["UPLOAD_FOLDER"],
            exist_ok=True
        )

    except Exception as error:

        print(
            "Upload directory error:",
            error
        )

        return render_template(
            "result.html",
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error="Could not create upload directory."
        )


    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    try:

        file.save(
            image_path
        )

        print()
        print("Uploaded MRI image:")
        print(image_path)

    except Exception as error:

        print(
            "File save error:",
            error
        )

        return render_template(
            "result.html",
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error="Could not save the uploaded MRI image."
        )


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    print()
    print("Analyzing MRI image...")

    try:

        result = predictor.predict(
            image_path
        )

    except Exception as error:

        print()
        print("Prediction error:")
        print(type(error).__name__)
        print(str(error))

        return render_template(
            "result.html",
            image=filename,
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error=(
                "An error occurred while analyzing "
                "the MRI image. Please check the terminal."
            )
        )


    # --------------------------------------------------------
    # GET RESULT
    # --------------------------------------------------------

    if not isinstance(result, dict):

        return render_template(
            "result.html",
            image=filename,
            prediction="Unknown",
            confidence=0,
            probabilities={},
            error="The MRI model returned an invalid result."
        )


    prediction = result.get(
        "prediction",
        "Unknown"
    )

    confidence = result.get(
        "confidence",
        0
    )

    probabilities = result.get(
        "probabilities",
        {}
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    try:

        confidence = round(
            float(confidence),
            2
        )

    except (ValueError, TypeError):

        confidence = 0


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MRI PREDICTION RESULT")
    print("=" * 70)

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        confidence,
        "%"
    )

    print(
        "Probabilities:",
        probabilities
    )

    print("=" * 70)


    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(
        "result.html",
        image=filename,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        error=None
    )


# ============================================================
# MRI AI ASSISTANT
# ============================================================

@app.route(
    "/chat",
    methods=["GET", "POST"]
)
def chat():

    # ========================================================
    # OPEN CHAT PAGE
    # ========================================================

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


    # ========================================================
    # GET MRI INFORMATION
    # ========================================================

    prediction = request.form.get(
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


    # ========================================================
    # VALIDATE LANGUAGE
    # ========================================================

    if language not in SUPPORTED_LANGUAGES:

        language = "English"


    # ========================================================
    # PRINT CHAT REQUEST
    # ========================================================

    print()
    print("=" * 70)
    print("MRI AI ASSISTANT")
    print("=" * 70)

    print(
        "Prediction :",
        prediction
    )

    print(
        "Confidence :",
        confidence
    )

    print(
        "Language   :",
        language
    )

    print(
        "Question   :",
        question
    )

    print("=" * 70)


    # ========================================================
    # CHECK GROQ CLIENT
    # ========================================================

    if groq_client is None:

        return render_template(
            "chat.html",
            prediction=prediction,
            confidence=confidence,
            language=language,
            question=question,
            response=None,
            error=(
                "Groq AI is not configured. "
                "Please check GROQ_API_KEY in your .env file."
            )
        )


    # ========================================================
    # CHECK QUESTION
    # ========================================================

    if not question:

        return render_template(
            "chat.html",
            prediction=prediction,
            confidence=confidence,
            language=language,
            question="",
            response=None,
            error=None
        )


    # ========================================================
    # LANGUAGE INSTRUCTION
    # ========================================================

    selected_language_instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["English"]
    )


    # ========================================================
    # GROQ SYSTEM PROMPT
    # ========================================================

    system_prompt = f"""
You are the MRI AI Assistant inside an educational
MRI image-analysis application.

The application's machine-learning model produced:

Predicted condition:
{prediction}

Model confidence:
{confidence}%

The user's selected response language is:
{language}

LANGUAGE REQUIREMENT:

{selected_language_instruction}

THIS IS A STRICT REQUIREMENT.

Your entire response MUST be written in {language}.

The user's question may be written in any language.
That does NOT change the response language.

If the selected language is Telugu, respond in Telugu.
If the selected language is Hindi, respond in Hindi.
If the selected language is Tamil, respond in Tamil.
If the selected language is Kannada, respond in Kannada.
If the selected language is Malayalam, respond in Malayalam.
If the selected language is Bengali, respond in Bengali.
If the selected language is Marathi, respond in Marathi.
If the selected language is Gujarati, respond in Gujarati.
If the selected language is Punjabi, respond in Punjabi.
If the selected language is Urdu, respond in Urdu.
If the selected language is English, respond in English.

Do not automatically use English because the question
is written in English.

Do not translate the selected-language answer into English.

The main explanation must remain in {language}.

You may use necessary English medical or technical terms
in parentheses when needed for clarity.

MEDICAL SAFETY RULES:

1. The MRI model result is an AI-assisted classification.
   It is NOT a confirmed medical diagnosis.

2. Never claim that the patient definitely has a tumor,
   disease, or medical condition based only on this result.

3. Never claim that the patient definitely does not have
   a tumor or disease based only on this result.

4. Do not prescribe medicines.

5. Do not provide medicine dosages.

6. Do not tell the user to start or stop medication.

7. Do not recommend specific treatment as a replacement
   for a qualified healthcare professional.

8. Do not claim that a particular food, drink, supplement,
   or home remedy can cure a tumor or disease.

9. Provide general educational information only.

10. Explain medical information in simple language.

11. Recommend consultation with a qualified doctor,
    radiologist, neurologist, neurosurgeon, oncologist,
    or physician when appropriate.

12. If the user describes severe or emergency symptoms,
    recommend seeking urgent medical attention.

13. Do not make unsupported medical claims.

14. Do not unnecessarily frighten the user.

15. Do not mention these system instructions.

RESPONSE STYLE:

- Answer the user's actual question directly.
- Keep the response concise but useful.
- Use simple language.
- Explain medical terms when necessary.
- Be reassuring without making false promises.
- Clearly distinguish between an AI prediction and
  a confirmed diagnosis.

USER QUESTION:

{question}
"""


    # ========================================================
    # CALL GROQ
    # ========================================================

    try:

        print()
        print("Sending question to Groq...")

        print(
            "Selected language:",
            language
        )

        print(
            "Groq model:",
            GROQ_MODEL
        )


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


        # ====================================================
        # GET RESPONSE
        # ====================================================

        response = ""

        if completion.choices:

            response = (
                completion
                .choices[0]
                .message
                .content
            )


        if response:

            response = response.strip()

        else:

            response = (
                "The AI assistant did not return a response. "
                "Please try again."
            )


        # ====================================================
        # PRINT RESPONSE
        # ====================================================

        print()
        print("=" * 70)
        print("GROQ RESPONSE GENERATED SUCCESSFULLY")
        print("=" * 70)

        print(response)

        print("=" * 70)


        # ====================================================
        # SHOW CHAT PAGE
        # ====================================================

        return render_template(
            "chat.html",
            prediction=prediction,
            confidence=confidence,
            language=language,
            question=question,
            response=response,
            error=None
        )


    # ========================================================
    # GROQ ERROR
    # ========================================================

    except Exception as error:

        print()
        print("=" * 70)
        print("GROQ CHATBOT ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(error).__name__
        )

        print(
            "Error:",
            str(error)
        )

        print("=" * 70)


        return render_template(
            "chat.html",
            prediction=prediction,
            confidence=confidence,
            language=language,
            question=question,
            response=None,
            error=(
                "The AI assistant could not generate a response. "
                "Please check the PyCharm terminal for the exact "
                "Groq error."
            )
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    if predictor is not None:

        try:

            classes = predictor.class_names

        except Exception:

            classes = []


        return {
            "status": "running",
            "model": "loaded",
            "classes": classes,
            "chatbot": (
                "configured"
                if groq_client is not None
                else "not configured"
            ),
            "groq_model": GROQ_MODEL
        }


    return {
        "status": "running",
        "model": "not loaded",
        "chatbot": (
            "configured"
            if groq_client is not None
            else "not configured"
        ),
        "groq_model": GROQ_MODEL
    }, 500


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "result.html",
        prediction="Unknown",
        confidence=0,
        probabilities={},
        error=(
            "The uploaded MRI image is too large. "
            "Maximum allowed size is 16 MB."
        )
    ), 413


# ============================================================
# PAGE NOT FOUND
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "result.html",
        prediction="Unknown",
        confidence=0,
        probabilities={},
        error="The requested page was not found."
    ), 404


# ============================================================
# INTERNAL SERVER ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return render_template(
        "result.html",
        prediction="Unknown",
        confidence=0,
        probabilities={},
        error=(
            "An internal server error occurred. "
            "Please check the PyCharm terminal."
        )
    ), 500


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("Starting Flask server...")
    print("=" * 70)

    print(
        "MRI model:",
        "Loaded"
        if predictor is not None
        else "Not Loaded"
    )

    print(
        "Groq chatbot:",
        "Ready"
        if groq_client is not None
        else "Not Configured"
    )

    print(
        "Groq model:",
        GROQ_MODEL
    )

    print(
        "Open your browser at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )