"""
Data generation script for PrimeRobotics fine-tuning dataset.

This module reads an input JSON dataset of question-answer pairs,
uses Google's Generative AI API to create rephrased variations,
and writes the expanded dataset to an output JSON file.

Environment Variables
---------------------
GENAI_API_KEY : str
    API key for Google Generative AI.
"""

from google import genai
from google.genai import errors
import os
import dotenv
from pydantic import BaseModel
import json
import time
from typing import List, Dict, Any
import traceback


JSON_INPUT_FILE: str = "primerobotics.json"
JSON_INPUT_FILE_OLD: str = "prime_robotics_old.json"
JSON_OUTPUT_FILE: str = "generated_primerobotics.json"


class PrimeRobotics(BaseModel):
    """
    Schema representing a single PrimeRobotics question-answer pair.

    Attributes
    ----------
    question : str
        The input question.
    answer : str
        The corresponding answer.
    """

    question: str
    answer: str


class GeneratedData(BaseModel):
    """
    Schema representing generated data response from the model.

    Attributes
    ----------
    generated_data : List[PrimeRobotics]
        A list of generated question-answer variations.
    """

    generated_data: List[PrimeRobotics]


def generate_data(
    question: str,
    answer: str,
    client: genai.Client,
) -> List[Dict[str, Any]]:
    """
    Generate rephrased variations of a question-answer pair using
    Google Generative AI.

    Parameters
    ----------
    question : str
        The original question.
    answer : str
        The original answer.
    client : genai.Client
        Initialized Google Generative AI client.

    Returns
    -------
    List[Dict[str, Any]]
        A list of generated question-answer dictionaries.

    Raises
    ------
    errors.ServerError
        If the API encounters a server-side error.
    json.JSONDecodeError
        If the response cannot be parsed as JSON.
    Exception
        For unexpected runtime errors.
    """
    prompt: str = """
        I want to fine-tune a model using the following data:
        question: {}
        answer: {}
        I have small data, so help me generate more data by 
        rephrasing the question and answer in different ways. 
        Generate at least 5 variations for each question and answer pair.
    """.format(question, answer)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": GeneratedData.model_json_schema(),
        },
    )

    gen_text: str = response.text
    parsed_response: Dict[str, Any] = json.loads(gen_text)

    return parsed_response["generated_data"]


def main() -> None:
    """
    Main execution function.

    Workflow
    --------
    1. Loads environment variables.
    2. Initializes the Google Generative AI client.
    3. Reads input JSON datasets.
    4. Iteratively generates additional data variations.
    5. Handles API and runtime errors with retry logic.
    6. Writes combined generated and old data to output JSON file.

    Notes
    -----
    - Stops execution after 5 consecutive server errors.
    - Sleeps between requests to prevent rate limiting.
    """
    dotenv.load_dotenv()

    client: genai.Client = genai.Client(
        api_key=os.getenv("GENAI_API_KEY")
    )

    start_index: int = 2
    error_count: int = 0
    generated_data: List[Dict[str, Any]] = []

    try:
        with open(JSON_INPUT_FILE, "r") as file:
            data: List[Dict[str, Any]] = json.load(file)
        size: int = len(data)

        with open(JSON_INPUT_FILE_OLD, "r") as file:
            old_data: List[Dict[str, Any]] = json.load(file)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    while start_index < size:
        question: str = data[start_index]["question"]
        answer: str = data[start_index]["answer"]

        try:
            gen_data: List[Dict[str, Any]] = generate_data(
                question, answer, client
            )
            generated_data.extend(gen_data)
            print(
                f"Generated data for index {start_index} added successfully."
            )

        except errors.ServerError as e:
            error_count += 1
            if error_count > 5:
                print(f"Too many errors. Stopped at index: {start_index}")
                print(f"Error: {e} - {type(e).__name__}")
                traceback.print_exc()
                break
            time.sleep(5)

        except Exception as e:
            error_type: str = type(e).__name__
            tb = traceback.extract_tb(e.__traceback__)
            _, line, *_ = tb[-1]
            print(f"{error_type}: {e} occurred at line {line}")
            break

        else:
            error_count = 0
            start_index += 1
            time.sleep(1)

    generated_data += old_data

    with open(JSON_OUTPUT_FILE, "w") as file:
        json.dump(generated_data, file, indent=4)


if __name__ == "__main__":
    main()
