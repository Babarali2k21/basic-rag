from openai import OpenAI

def get_openai_embedding(text, client=None):
    if client is None:
        client = OpenAI()
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding