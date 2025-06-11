from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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

# Set up retriever (limit to 3 most relevant chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load FLAN-T5-small model locally (free and instruction-tuned)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Hugging Face pipeline for text2text-generation
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def deduplicate_sentences(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        norm = re.sub(r'[^a-zA-Z0-9\s]', '', sentence.lower())
        if norm not in seen:
            seen.add(norm)
            unique_sentences.append(sentence)
    return ' '.join(unique_sentences)

def is_nonsense(text):
    nonsense_phrases = [
        "colds can develop in blood",
        "colds frequently develop under normal skin conditions",
        "heart disease can be caused by eating fruits"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in nonsense_phrases)

def clean_response(text):
    text = re.sub(r'Toni Rizzo|Dallas|Houston', '', text, flags=re.IGNORECASE)
    text = ' '.join(text.split())
    text = deduplicate_sentences(text)
    return text

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
        context = " ".join([doc.page_content for doc in docs])[:2000]  # Limit context length to avoid cutoff
        
        prompt = (
            "You are a helpful medical assistant. Use the medical information from the context below to answer the user's question. "
            "If the answer is not found in the context, say 'I don't know.' Keep the answer short and factual.\n\n"
            f"Context: {context}\n\n"
            f"Question: {user_input}\n"
            "Answer:"
        )

        response = qa_pipeline(
            prompt,
            max_new_tokens=400,
            do_sample=False
        )[0]['generated_text']

        cleaned_response = response.strip()
        cleaned_response = clean_response(cleaned_response)
        if is_nonsense(cleaned_response):
            cleaned_response = "Sorry, I couldn't provide a reliable answer. Please consult a healthcare provider."
        return jsonify({"response": cleaned_response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Sorry, there was an error processing your request."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
