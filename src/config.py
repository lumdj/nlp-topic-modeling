import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# Project settings
PROJECT_NAME = os.getenv("PROJECT_NAME", "nlp-topic-pipeline")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validate key exists
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in .env file")
