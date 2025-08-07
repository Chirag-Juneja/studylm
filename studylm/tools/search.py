from langchain_community.tools import DuckDuckGoSearchResults


search = DuckDuckGoSearchResults(
    regoin="in-en", max_results=5, output_format="list")
