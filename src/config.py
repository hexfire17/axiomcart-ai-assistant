"""
Configuration: Logger, API clients, and settings.

This module is imported by everything else, so it must have zero
dependencies on other project modules.
"""

import logging
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

# ── Logger ───────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format and file output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Define the common format for both console and file
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S"
        )

        # 1. Console Handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler (application.log)
        # 'a' mode ensures it appends to the file rather than overwriting it
        file_handler = logging.FileHandler("application.log", mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optional: prevent logs from doubling up if the root logger is configured
        logger.propagate = False

    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("config")

# ── Validate API key ────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
    logger.error("OPENAI_API_KEY is missing. Copy .env.example → .env and add your key.")
    sys.exit(1)

# ── Clients ──────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

logger.info("OpenAI clients initialised  (model: gpt-4o, embeddings: text-embedding-3-small)")
