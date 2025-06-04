from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pymongo
import traceback
import re

app = Flask(__name__)
CORS(app)

# URLs and credentials
OLLAMA_URL = "http://192.168.1.117:11434/api/generate"
MODEL_NAME = "optgpt:7b"

MONGO_URI = "mongodb+srv://hriday4304:sDwCr1PjI6cYOH29@cluster0.kyzbcod.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["sentiment_rag"]
collection = db["documents"]

# Optional RAG document retrieval (not used in logic yet)
def retrieve_documents(query, top_n=3):
    docs = collection.find()
    scored = []
    for doc in docs:
        content = doc.get('content', '').lower()
        score = sum([1 for word in query.lower().split() if word in content])
        if score > 0:
            scored.append((score, doc.get('content', '')))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored][:top_n]

@app.route("/api/sentiment", methods=["POST"])
def sentiment():
    try:
        data = request.get_json()
        review = data.get("review", "").strip()
        if not review:
            return jsonify({"error": "Review text is required."}), 400

        # Document retrieval for future RAG use
        _ = retrieve_documents(review)

        # Strict prompt to enforce single word output
        prompt = f"""
You are a sentiment classification AI. Analyze the sentiment of the following movie review.
Respond with ONLY one of these words (case-insensitive): Positive, Negative, or Neutral.
Do not include any explanation, punctuation, or additional text.

Review: {review}
"""

        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }, timeout=60)

        if response.status_code != 200:
            return jsonify({
                "error": "LLM Error",
                "llm_status": response.status_code,
                "llm_details": response.text,
                "prompt_sent": prompt
            }), 500

        llm_response = response.json().get("response", "").strip().lower()
        print(f"[DEBUG] Raw LLM response: {llm_response}")

        # Use regex to extract the correct word
        match = re.search(r"\b(positive|negative|neutral)\b", llm_response)
        if match:
            sentiment_label = match.group(1)
        else:
            sentiment_label = "neutral"  # fallback

        return jsonify({
            "review": review,
            "sentiment": sentiment_label
        })

    except Exception as e:
        return jsonify({
            "error": "Internal Server Error",
            "details": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
