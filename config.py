from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )

    # RAG
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K")

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./chroma_db", alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection: str = Field(
        default="documents", alias="CHROMA_COLLECTION"
    )

    # Articles
    articles_dir: str = Field(default="./articles", alias="ARTICLES_DIR")


settings = Settings()
