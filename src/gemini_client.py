import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )

client = genai.Client(api_key=API_KEY)


def generate_answer(prompt, max_retries=3):
    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            return response.text

        except errors.ServerError as e:

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)

            else:
                return (
                    "Gemini is temporarily unavailable due to "
                    "high demand. Please try again in a few moments."
                )

        except Exception as e:

            return f"Gemini API error: {str(e)}"


if __name__ == "__main__":
    answer = generate_answer(
        "Reply with exactly: Gemini connection successful."
    )

    print(answer)