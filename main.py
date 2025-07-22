from studylm.utils.logger import get_logger
from studylm.core.llm import get_graph_transformer
from studylm.core.pdf_parser import parse_pdf

logger = get_logger(__name__)

path = "./data/AI Engineering.pdf"
transformer = get_graph_transformer()
chunks = parse_pdf(path)
graph_docs = transformer.convert_to_graph_documents(chunks[:1])

logger.info(graph_docs)
