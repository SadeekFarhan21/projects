"""Central configuration using pydantic-settings"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    nessie_api_key: str = "your_key_here"
    kafka_bootstrap_servers: str = "localhost:19092"
    ollama_base_url: str = "http://localhost:11434"
    mock_mode: bool = True

    # Transaction generator settings
    txn_interval_min: float = 1.0
    txn_interval_max: float = 3.0
    anomaly_probability: float = 0.08  # 8% chance of anomaly per transaction

    # Anomaly detection
    anomaly_threshold: float = -0.3  # IsolationForest decision function threshold

    # Agent settings
    ollama_model: str = "llama3.2:3b"  # Smaller/faster; set OLLAMA_MODEL=llama3:8b for higher quality

    # Dispute settings
    dispute_probability: float = 0.03  # 3% of transactions get disputed

    # FAISS vector store (local, in-memory — no API keys needed)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # Neo4j graph database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = ""
    neo4j_password: str = ""

    # Redis cache
    redis_url: str = "redis://localhost:6379/0"

    # Webhook (Slack-compatible, leave empty to disable)
    webhook_url: str = ""          # AI agent BLOCK verdicts → #fraud-alerts
    webhook_url_human: str = ""    # Human HITL BLOCK decisions → #human-reviews

    # Demo mode
    demo_mode: bool = False
    demo_txn_interval: float = 6.0
    demo_anomaly_interval: float = 30.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
