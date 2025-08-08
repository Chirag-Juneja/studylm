from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from studylm.tools.search import search
from studylm.tools.arxiv import arxiv_search
from studylm.tools.deepresearch import deepresearch
from studylm.config.constants import MODEL


def get_reasearch_agent():
    llm = ChatOllama(model=MODEL, temperature=0.1, max_tokens=5000)
    research_agent = create_react_agent(
        model=llm,
        tools=[search, arxiv_search, deepresearch],
        prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_agent"
    )
    return research_agent
