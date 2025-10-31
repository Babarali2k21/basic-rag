from openai import OpenAI

def generate_response(question, relevant_chunks, client=None):
    if client is None:
        client = OpenAI()
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following context to answer concisely:\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message