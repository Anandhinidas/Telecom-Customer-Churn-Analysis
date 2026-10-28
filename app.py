from flask import Flask, render_template, request, jsonify
from ai_query_assistant import ask_churn_assistant

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."})

    try:
        answer = ask_churn_assistant(question)
    except Exception as e:
        answer = f"Something went wrong: {str(e)}"

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)