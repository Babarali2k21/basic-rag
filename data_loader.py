import os

class DataLoader:
    def __init__(self, chunk_size=1000, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_from_folder(self, folder_path):
        docs = []
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(folder_path, file), "r", encoding="utf8") as f:
                    docs.append({"id": file, "text": f.read()})
        return docs

    def spllit_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.overlap
