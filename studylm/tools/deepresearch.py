from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool
from studylm.tools.search import search
from langchain_community.document_loaders import WebBaseLoader


@tool
def deepresearch(query: str):
    """
    Conduct comprehensive deep research on the internet by:
        1. Getting duckduckgo search results
        2. Crawling each web link to extract full content
        3. Synthesizing all information together
    Use this tool for complex topics requiring through analysis from multiple sources. 
    """
    search_links = DuckDuckGoSearchResults(
        regoin="in-en", max_results=5, output_format="list")

    search_response = search.invoke(query)

    links = []
    for response in search_links.invoke(query):
        links.append(response["link"])

    loader = WebBaseLoader(web_paths=links)

    texts = []
    for doc in loader.lazy_load():
        texts.append(doc.page_content)
    return "\n\n---\n\n".join([search_response]+texts)

