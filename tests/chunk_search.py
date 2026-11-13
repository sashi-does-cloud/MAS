import os
import numpy as np
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


def load_documents():

    documents = []
    path = os.path.join("knowledge_base")

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

def get_embedding(source_text):

    response = client.embeddings.create(
        input=source_text,
        model=os.getenv("EMBED_MODEL"),
    )

    return response.data[0].embedding

def add_embeddings(chunks):

    store_path = os.path.join("store.json")

    try:
        with open(store_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if data is not None:
                return data 

    except FileNotFoundError:
        for chunk in chunks:
            embedding = get_embedding(chunk['text'])
            chunk["embedding"] = embedding

        with open(store_path, "w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=4)
        
        return chunks

def cosine_similarity(A, B):

    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)

    score = score = float(np.dot(A, B) / (norm_a * norm_b))

    return score

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


def make_chunks(documents):

    chunks = []

    for document in documents:
        chunk = get_chunk(document)
        chunks.extend(chunk)
    
    return chunks


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

    # documents_with_scores.sort(*, score: lambda, reverse=True)

    sorted_chunks = sorted(res, key=lambda x: x['score'], reverse=True)
    print(json.dumps(sorted_chunks[:top_k], indent=3))


    


documents = load_documents()
chunks = make_chunks(documents)
chunks = add_embeddings(chunks)

query("what are the benefits of maternity leave?", chunks, 3)

