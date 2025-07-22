from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from studylm.utils.logger import get_logger

logger = get_logger(__name__)


def get_graph_transformer(model="gemma3:4b"):
    logger.info(f"Loading {model} model")
    llm = ChatOllama(model=model, temperature=0.1, max_tokens=5000)
    transformer = LLMGraphTransformer(llm=llm)
    return transformer
