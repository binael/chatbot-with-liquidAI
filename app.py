from flask import Flask, render_template, request, jsonify
from chatbot import chatbot

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")

    reply = chatbot(user_message)

    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )

