from openai import OpenAI
from config import Config

config = Config()
client = OpenAI(api_key=config.openai_key)

def generate_response(question: str, relevant_chunks: list):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
    )
    return response.choices[0].message.content