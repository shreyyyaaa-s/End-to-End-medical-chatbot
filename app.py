from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load env vars
load_dotenv()

# Pinecone init
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# FLAN-T5 pipeline
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# --------- Utilities ---------
def deduplicate_sentences(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    seen = set()
    unique = []
    for s in sentences:
        norm = re.sub(r'[^a-zA-Z0-9\s]', '', s.lower())
        if norm not in seen:
            seen.add(norm)
            unique.append(s)
    return ' '.join(unique)

def clean_pdf_artifacts(text):
    # Fix broken words and weird line breaks from PDFs
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  # espe- cially → especially
    text = re.sub(r'\s+', ' ', text)                # normalize whitespace
    return text.strip()

def is_nonsense(text):
    junk = ["colds develop in blood", "heart disease from fruits", "colds under skin"]
    return any(j in text.lower() for j in junk)

def clean_response(text):
    text = re.sub(r'Toni Rizzo|Dallas|Houston', '', text, flags=re.IGNORECASE)
    text = ' '.join(text.split())
    return deduplicate_sentences(text)

# --------- Flask ---------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        user_input = request.form["message"]
        docs = retriever.get_relevant_documents(user_input)
        raw_context = " ".join([doc.page_content for doc in docs])
        context = clean_pdf_artifacts(raw_context)
        context = deduplicate_sentences(context)[:2000]

        prompt = (
            "You are a helpful medical assistant. Based on the context, answer the user's question factually and clearly. "
            "If the question is about a condition, first give a definition, then add symptoms or treatment if available. "
            "If the answer isn't in the context, say 'I don't know.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {user_input}\nAnswer:"
        )

        response = qa_pipeline(prompt, max_new_tokens=400, do_sample=False)[0]['generated_text']
        final_response = clean_response(response)
        if is_nonsense(final_response):
            final_response = "Sorry, I couldn't provide a reliable answer. Please consult a healthcare provider."

        return jsonify({"response": final_response})

    except Exception as e:
        print("Error:", e)
        return jsonify({"response": "Sorry, there was an error processing your request."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
