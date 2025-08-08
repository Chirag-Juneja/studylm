from langchain_community.document_loaders import ArxivLoader
from langchain.tools import tool


@tool
def arxiv_search(query):
    """
    Search for academic research papers from arXiv on a given topic by:

    1.Fetching full text of the research papers.

    Use this for complex topics requiring thorough analysis from academic sources
    """
    loader = ArxivLoader(query, load_max_docs=5)
    docs = loader.load()
    texts = []
    for doc in docs:
        texts.append(f"Title{doc.metadata.get("Title")},\n\n{doc.page_content}")
    return "\n\n---\n\n".join(texts)
