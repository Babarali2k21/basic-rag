import os

def load_documents_from_directory(directory_path: str):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text: str, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks