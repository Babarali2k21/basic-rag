from openai import OpenAI
from config import Config

config = Config()
client = OpenAI(api_key=config.openai_key)

def get_openai_embedding(text: str):
    response = client.embeddings.create(
        input=text,
        model=config.embedding_model
    )
    return response.data[0].embedding