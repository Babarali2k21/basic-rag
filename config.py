import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_PATH = "chroma_persistent_storage"
    COLLECTION_NAME = "document_qa_collection"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-3.5-turbo"