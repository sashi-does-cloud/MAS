import chromadb
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ingest import load_documents_with_index
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def get_embedding(source):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=source
    )
    return response.data[0].embedding

docs = load_documents_with_index()

ids = list(docs.keys())
documents = list(docs.values())

chroma_client = chromadb.Client()

zorvex_collection = chroma_client.create_collection(
    name="zorvex_technology",
    configuration={
        "hsnw": {
            "space": "cosine"
        }
    }
)

embeddings = []
for doc in documents:
    embeddings.append(get_embedding(doc))

zorvex_collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

query_embedding = get_embedding("when is production deploeyment is alllowed")

results = zorvex_collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(json.dumps(results, indent=4))


