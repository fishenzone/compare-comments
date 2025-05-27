from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None
    QDRANT_DISTANCE_METRIC: str = "Cosine"
    QDRANT_COLLECTION_V1_PREFIX: str = "doc_v1_"
    QDRANT_COLLECTION_V2_PREFIX: str = "doc_v2_"

    VLLM_HOST: str = "vllm"
    VLLM_PORT: int = 8000
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    VLLM_API_KEY: str = "secret"

    EMBEDDING_MODEL_NAME: str = "sergeyzh/BERTA"
    EMBEDDING_DIM: int = 768
    DEFAULT_DOC_PREFIX: str = "search_document: "
    DEFAULT_COMMENT_PREFIX: str = "search_query: "

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(env_file=None, extra="ignore")


settings = Settings()

QDRANT_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
VLLM_BASE_URL = f"http://{settings.VLLM_HOST}:{settings.VLLM_PORT}/v1"

print("--- Configuration Loaded ---")
print(f"Qdrant Host: {settings.QDRANT_HOST}")
print(f"Qdrant Port: {settings.QDRANT_PORT}")
print(f"vLLM Host: {settings.VLLM_HOST}")
print(f"vLLM Port: {settings.VLLM_PORT}")
print(f"vLLM Model: {settings.VLLM_MODEL}")
print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
print(f"Embedding Dim: {settings.EMBEDDING_DIM}")
print(f"Default Doc Prefix: '{settings.DEFAULT_DOC_PREFIX}'")
print(f"Default Comment Prefix: '{settings.DEFAULT_COMMENT_PREFIX}'")
