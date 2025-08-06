import faiss
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from studylm.config.constants import MODEL, FAISS_PATH
from langchain.tools.retriever import create_retriever_tool


class VectorStore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=MODEL)
        self.persist_path = FAISS_PATH
        self.faiss_index = faiss.IndexFlatL2(len(self.embeddings.embed_query("test")))
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=self.faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def as_retriever(self, k=5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def as_tool(self, k=5):
        return create_retriever_tool(
            self.as_retriever(k),
            "retrieve_context",
            "Search and get context based on documents shared by the user",
        )

    def add_documents(self, docs):
        self.vectorstore.add_documents(docs)
        self.vectorstore.save_local(self.persist_path)
