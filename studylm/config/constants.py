import logging
from pathlib import Path

# global variables
LOG_LEVEL = logging.INFO
MODEL = "qwen3:14b"
TEMP_PATH = "./temp/"
FAISS_PATH = TEMP_PATH + "faiss_index"
Path(TEMP_PATH).mkdir(parents=True, exist_ok=True)
