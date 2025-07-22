from studylm.utils.logger import get_logger
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = get_logger(__name__)


def parse_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Total number of chunks extracted: {len(chunks)}")
    return chunks
