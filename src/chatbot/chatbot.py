import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# CHECK API KEY
# ============================================================

if not GROQ_API_KEY:

    print("=" * 70)
    print("ERROR: GROQ_API_KEY was not found.")
    print("Make sure your .env file contains:")
    print("GROQ_API_KEY=your_groq_api_key")
    print("=" * 70)

    groq_client = None

else:

    print("=" * 70)
    print("Groq API key loaded successfully.")
    print("=" * 70)

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# GET AI RESPONSE
# ============================================================

def get_chat_response(
    question,
    prediction,
    confidence,
    language="English"
):

    # --------------------------------------------------------
    # CHECK QUESTION
    # --------------------------------------------------------

    if not question or not question.strip():

        return "Please enter a question."


    # --------------------------------------------------------
    # CHECK GROQ CLIENT
    # --------------------------------------------------------

    if groq_client is None:

        raise Exception(
            "GROQ_API_KEY is missing. "
            "Check your .env file."
        )


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = f"""
You are the MRI AI Assistant in an intelligent MRI
image analysis application.

The machine learning model analyzed an MRI image.

Prediction:
{prediction}

Confidence:
{confidence}%

User selected language:
{language}

Your job is to answer the user's question about the MRI
result in a clear, simple and helpful way.

IMPORTANT RULES:

1. Answer in the language selected by the user.
2. If the language is Telugu, answer in Telugu.
3. Explain medical concepts in simple language.
4. Do not claim that you are a doctor.
5. Do not give a definitive medical diagnosis.
6. Explain that the AI prediction must be confirmed by
   a qualified medical professional.
7. Answer the user's actual question directly.
8. Do not unnecessarily repeat the entire system prompt.
9. Keep the response reasonably concise.
"""


    # --------------------------------------------------------
    # CALL GROQ
    # --------------------------------------------------------

    try:

        print("\n" + "=" * 70)
        print("SENDING REQUEST TO GROQ")
        print("=" * 70)

        print("Question:", question)
        print("Prediction:", prediction)
        print("Confidence:", confidence)
        print("Language:", language)

        response = groq_client.chat.completions.create(

            model="llama-3.3-70b-versatile",

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

            temperature=0.3,

            max_completion_tokens=500
        )


        # ----------------------------------------------------
        # GET RESPONSE TEXT
        # ----------------------------------------------------

        answer = response.choices[0].message.content


        print("\nGROQ RESPONSE RECEIVED")
        print("=" * 70)
        print(answer)
        print("=" * 70)


        return answer


    # --------------------------------------------------------
    # SHOW ACTUAL ERROR
    # --------------------------------------------------------

    except Exception as error:

        print("\n" + "=" * 70)
        print("GROQ API ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(error).__name__
        )

        print(
            "Error message:",
            str(error)
        )

        print("=" * 70)


        # IMPORTANT:
        # Raise the error so app.py can display it.

        raise