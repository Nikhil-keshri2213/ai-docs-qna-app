import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "DocMind AI"
    # Optional: only needed if you want to use HuggingFace Hub API models
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

settings = Settings()