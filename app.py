from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalchatbot"

# Create index if not exists
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector store
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="text"
)

# Set up retriever
retriever = vectorstore.as_retriever()

# HF Transformers pipeline
qa_pipeline = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

def deduplicate_sentences(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    return ' '.join(unique_sentences)

def is_nonsense(text):
    # Example: Check for common errors or nonsense phrases
    nonsense_phrases = [
        "colds can develop in blood",
        "colds frequently develop under normal skin conditions"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in nonsense_phrases)

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        user_input = request.form["message"]
        docs = retriever.get_relevant_documents(user_input)
        context = " ".join([doc.page_content for doc in docs])
        prompt = (
            "You are a medical assistant. Use only the factual medical information from the context below to answer the question. "
            "Ignore any personal stories, names, or locations. "
            "Do not repeat information. "
            "If you don't know the answer, say you don't know. "
            "Keep your answer concise and to the point (no more than three sentences).\n\n"
            f"Context: {context}\n\n"
            f"Question: {user_input}\n"
            "Answer:"
        )
        response = qa_pipeline(
            prompt,
            max_new_tokens=100,
            truncation=True,
            do_sample=True
        )[0]['generated_text']
        cleaned_response = response.replace(prompt, "").strip()
        # Further clean the response (remove names, locations, extra spaces)
        cleaned_response = re.sub(r'Toni Rizzo|Dallas|Houston', '', cleaned_response)
        cleaned_response = ' '.join(cleaned_response.split())
        # Deduplicate sentences
        cleaned_response = deduplicate_sentences(cleaned_response)
        # Check for nonsense
        if is_nonsense(cleaned_response):
            cleaned_response = "Sorry, I couldn't provide a reliable answer. Please consult a healthcare provider."
        return jsonify({"response": cleaned_response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Sorry, there was an error processing your request."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
