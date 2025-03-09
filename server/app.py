from flask import Flask, request, jsonify
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

app = Flask(__name__)

llm = HuggingFacePipeline.from_model_id("HuggingFaceH4/zephyr-7b-alpha")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local("faiss_index", embeddings).as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

def process_document(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    return texts

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    response = qa_chain.run(question)
    return jsonify({"answer": response})

@app.route("/add_document", methods=["POST"])
def add_document():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    texts = process_document(filepath)
    retriever.vectorstore.add_documents(texts)
    return jsonify({"message": "Document added successfully"})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)