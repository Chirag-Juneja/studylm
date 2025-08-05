from studylm.utils.logger import get_logger
from langchain_community.document_loaders import PDFPlumberLoader

logger = get_logger(__name__)


def parse_pdf(file_path: str):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    logger.info(f"Total number of pages extracted:{len(documents)}")
    return documents
