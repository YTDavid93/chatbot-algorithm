from flask import Flask, render_template, request, jsonify
import json
from flask_cors import CORS
from train import TFIDFVectorizer

app = Flask(__name__)
CORS(app)


# Initialize TFIDFVectorizer instance
vectorizer = TFIDFVectorizer()

with open('intents.json', 'r', encoding="utf-8") as file:
    intents = json.load(file)

patternList = []
answers = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patternList.append(pattern)
        answers.append(intent["responses"][0])

# Set the patterns and answers for vectorizer
vectorizer.answers = answers
total_corpus, doc_a = vectorizer.build_corpus(patternList)
vectorizer.calculate_idf(doc_a)
vectorizer.precompute_document_vectors()

@app.route("/")
def index():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("question")
    # Check if JSON is valid
    response = vectorizer.get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__=="__main__":
    app.run(debug=True)