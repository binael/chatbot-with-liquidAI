"""
Flask application serving the Prime Robotics fine-tuned chatbot.

This module exposes two routes:

- "/" (GET)
    Renders the chatbot web interface.

- "/ask" (POST)
    Accepts a JSON payload containing a user message,
    forwards it to the fine-tuned chatbot model,
    and returns the generated response as JSON.

The chatbot logic is imported from `chatbot.py`.

Run
---
python app.py

Server
------
Host: 0.0.0.0
Port: 5000
"""

from typing import Any, Dict

from flask import Flask, render_template, request, jsonify, Response
from chatbot import chatbot


app: Flask = Flask(__name__)


@app.route("/")
def index() -> str:
    """
    Render the chatbot homepage.

    Returns
    -------
    str
        Rendered HTML template for the chatbot interface.
    """
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask() -> Response:
    """
    Handle chatbot interaction requests.

    Expects
    -------
    JSON body:
        {
            "message": str
        }

    Returns
    -------
    flask.Response
        JSON response containing:
        {
            "response": str
        }

    Raises
    ------
    KeyError
        If "message" field is missing in the request body.
    Exception
        If the chatbot model raises an error during inference.
    """
    data: Dict[str, Any] = request.get_json(force=True)
    user_message: str = data.get("message", "")

    reply: str = chatbot(user_message)

    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
