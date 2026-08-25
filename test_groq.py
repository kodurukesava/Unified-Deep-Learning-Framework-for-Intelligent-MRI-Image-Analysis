import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")

print("API KEY FOUND:", bool(api_key))

if not api_key:
    print("GROQ_API_KEY not found")
    exit()

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Hello, explain MRI in one sentence."
            }
        ],
        max_completion_tokens=100
    )

    print("\nSUCCESS!")
    print(response.choices[0].message.content)

except Exception as error:
    print("\nGROQ ERROR!")
    print(type(error).__name__)
    print(error)