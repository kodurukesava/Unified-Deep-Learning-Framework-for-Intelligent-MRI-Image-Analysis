from .safety import SAFETY_RULES


def create_prompt(
    language,
    prediction,
    confidence,
    user_question
):

    prompt = f"""
{SAFETY_RULES}

The MRI analysis system produced the following AI result:

Predicted class:
{prediction}

Model confidence:
{confidence}%

Preferred response language:
{language}

User question:
{user_question}

Instructions:

1. Respond in the requested language.
2. Explain the result in simple language.
3. Clearly say that the AI result is NOT a confirmed medical diagnosis.
4. Do not invent patient-specific medical facts.
5. For food questions, provide only general supportive nutrition
   information and advise consultation with a qualified doctor or
   dietitian for individualized recommendations.
6. For precaution questions, provide general precautions and
   follow-up considerations.
7. If appropriate, suggest what the user can discuss with a doctor.
8. Do not prescribe medicines or treatment.
"""

    return prompt