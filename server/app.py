from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")

generator = pipeline("text-generation", model="openai-community/gpt2", api_key=API_KEY)

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = generator(question, max_length=100, num_return_sequences=1)
        generated_text = response[0]['generated_text']
        return jsonify({'response': generated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)