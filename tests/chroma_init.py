import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
import json
from dotenv import load_dotenv
from utils.ingest import load_documents


client = chromadb.Client()

load_dotenv()

zorvex_collection = client.create_collection(
    name="zorvex-technology"
)

DOC_FOLDER = os.getenv("DOC_FOLDER")

documents = load_documents()

zorvex_collection.add(
    ids=["travel_expense", "leave_policy", "medical_policy", "hr_docs", "tech_docs"],
    documents=documents
)

results = zorvex_collection.query(
    query_texts=["when is the production deployment allowed."],
    n_results=1
)

print(json.dumps(results, indent=4))