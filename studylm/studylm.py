import faiss
import re
from langchain_ollama import ChatOllama
from studylm.utils.logger import get_logger
from studylm.config.constants import MODEL
from studylm.core.pdf_parser import parse_pdf
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


logger = get_logger(__name__)


def remove_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class StudyLM:
    def __init__(self):
        self.llm = ChatOllama(model=MODEL, extract_reasoning=False)
        self.embeddings = OllamaEmbeddings(model=MODEL)
        self.uploaded_docs = []
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.faiss_index = faiss.IndexFlatL2(len(self.embeddings.embed_query("test")))
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=self.faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
            verbose=True,
        )

    def upload_document(self, file_path: str, filename: str):
        self.uploaded_docs.append(filename)
        docs = parse_pdf(file_path)
        self.vectorstore.add_documents(docs)
        return True
