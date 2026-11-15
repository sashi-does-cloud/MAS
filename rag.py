import os
import numpy as np
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DOC_FOLDER = os.getenv("DOC_FOLDER")
EMBED_MODEL = os.getenv("EMBED_MODEL")
VECTOR_STORE = os.getenv("VECTOR_STORE")
STORE_PATH = os.path.join(VECTOR_STORE)
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Load Documents from DOC_FOLDER
def load_documents():

    documents = []
    path = os.path.join(DOC_FOLDER)

    for id, filename in enumerate(os.listdir(path)):
        file_path = os.path.join(path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read()
            documents.append({
                "id": id,
                "source": filename,
                "text": data
            })
    
    return documents

# Get embedding for source text
def get_embedding(source_text):

    response = client.embeddings.create(
        input=source_text,
        model=EMBED_MODEL,
    )

    return response.data[0].embedding

# Add embeddings to chunks
def add_embeddings(chunks):

    for chunk in chunks:
        embedding = get_embedding(chunk['text'])
        chunk["embedding"] = embedding
        
    return chunks

# Calculate cosine similarity
def cosine_similarity(A, B):

    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    score = score = float(np.dot(A, B) / (norm_a * norm_b))

    return score

# Split document into chunks
def get_chunk(document):

    chunks = []
    for chunk_id, section in enumerate(document['text'].split("\n\n")):
        section = section.strip()
        if section:

            chunks.append({
                "source": document['source'],
                "chunk_id": chunk_id,
                "text": section

            })

    return chunks

# Make chunks from documents
def make_chunks(documents):

    chunks = []
    for document in documents:
        chunk = get_chunk(document)
        chunks.extend(chunk)

    return chunks

# Save Embeddings
def save_vector_store(chunks):

    with open(STORE_PATH, "w") as file:
        file.write(json.dumps(chunks, indent=3))

# Load Embeddings
def load_vector_store():

    with open(STORE_PATH, "r") as file:
        data = json.load(file)
        return data

# Load Chunks
def load_chunks():

    chunks = []
    if os.path.exists(STORE_PATH):
        chunks = load_vector_store()
        print(f"Loaded {len(chunks)} chunks from {STORE_PATH}")
        
    else:
        documents = load_documents()
        chunks = make_chunks(documents)
        chunks = add_embeddings(chunks)

        save_vector_store(chunks)

        print(f"Created {len(chunks)} chunks from {DOC_FOLDER}")
    
    return chunks

# Search chunks using query
def query(prompt, chunks, top_k):

    embedding = get_embedding(prompt)
    res = []
    for chunk in chunks:
        cosine = cosine_similarity(embedding, chunk["embedding"])
        res.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "score": cosine
        })

    sorted_chunks = sorted(res, key=lambda x: x['score'], reverse=True)
    print(json.dumps(sorted_chunks[:top_k], indent=3))


