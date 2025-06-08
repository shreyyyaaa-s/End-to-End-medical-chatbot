from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please set it in your .env file.")

# Load documents
documents = load_pdf_file(r"C:\Users\balam\Documents\End-to-End-medical-chatbot\Data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
text_chunks = text_split(documents)

# Get embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_328Qu2_67nfjyvbXVqxTMT6JAKptWNThuYtqr1ixdziPazTSWMN2pibc8Fi5p8i2PsQXrX")

# Define index name
index_name = "medicalchatbot"

# Create index if not exists
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Store embeddings into Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
