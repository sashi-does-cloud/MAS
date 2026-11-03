from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

res = client.embeddings.create(
    model=os.getenv("EMBED_MODEL"),
    input=["I am a good boy", "he is a good coder"]
)

for i, emb in enumerate(res.data):
    print(f"Embedding {i} : {emb.embedding[:5]}")