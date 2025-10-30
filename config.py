import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-3.5-turbo"
        self.chroma_path = "chroma_persistent_storage"
        self.collection_name = "document_qa_collection"