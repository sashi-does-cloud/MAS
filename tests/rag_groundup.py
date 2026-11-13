import os
import numpy as np
import json

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

DOC_FOLDER = os.getenv("DOC_FOLDER")

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def load_documents(folder=os.getenv("DOC_FOLDER")):
    documents = []
    for filename in sorted(os.listdir(folder)):
        
        path = os.path.join(folder, filename)
        
        with open(path, "r", encoding="utf-8") as f:
            documents.append({
                "source": filename,
                "text": f.read(),
            })
        
    return documents

def get_embedding(text):

    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=text
    )

    return response.data[0].embedding

def cosine_similarity(a, b):

    # print(type(b), type(a))


    dot_prod = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_prod / (norm_a * norm_b)
    
def add_embeddings(documents):

    for doc in documents:
        embed = get_embedding(doc["text"])
        doc["embedding"] = embed
    
    return documents


def search(query, documents, top_k):

    results = []
    query_embedding = get_embedding(query)

    for doc in documents:
        score = float(cosine_similarity(doc["embedding"], query_embedding))
        results.append({
            "score": score,
            "source": doc["source"],
            "text": doc["text"],
            "size": len(doc["text"])
        })

    results.sort(key=lambda doc: doc["score"], reverse=True)
    
    return results[:top_k]


def test(query, documents, k):

    print(f"user query: {query}")

    results = search(query, documents, k)

    for r in results:
        print(30 * "*")
        print(f"score: {r['score']:.2f} | source: f{r['source']} | text: f{r['text']}")



documents = load_documents()
documents = add_embeddings(documents)
# print(json.dumps(documents, indent=4))

test("how many days will i get for maternity leave", documents, 3)



