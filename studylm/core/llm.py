from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama
from studylm.utils.logger import get_logger
from studylm.config.constants import MODEL

logger = get_logger(__name__)


llm = ChatOllama(model=MODEL, temperature=0.1, max_tokens=5000)
transformer = LLMGraphTransformer(llm=llm)
