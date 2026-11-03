import os
import numpy as np

from openai import OpenAI
from dotenv import load_dotenv

def cosine_similarity(a, b):

    dot_prod = np.dot(a, b)
    mod_mult = np.linalg.norm(a) * np.linalg.norm(b)

    return dot_prod / mod_mult


load_dotenv()    


client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

load = [
    "search_document: i love python since it is very intuitive",
    "search_document: i love python since it is venomous"
]


response = client.embeddings.create(
    model=os.getenv("EMBED_MODEL"),
    input=load
)

embed1 = response.data[0].embedding
embed2 = response.data[1].embedding

print(cosine_similarity(embed1, embed2))





