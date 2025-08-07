import uuid
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from studylm.utils.logger import get_logger
from studylm.config.constants import MODEL
from studylm.core.pdf_parser import parse_pdf
from studylm.tools.rag import VectorStore
from langchain.globals import set_verbose
from typing import TypedDict, List, Optional, Annotated
from studylm.tools.youtube import yt_search
from studylm.tools.arxiv import arxiv_search
from studylm.tools.search import search

set_verbose(True)

logger = get_logger(__name__)


class GraphState(TypedDict):
    messages: Annotated[List, add_messages]
    yt_links: Optional[List]


class StudyLM:
    def __init__(self):
        self.llm = init_chat_model(model=MODEL, model_provider="ollama")
        self.config = {"configurable": {"thread_id": uuid.uuid4()}}
        self.uploaded_docs = []
        self.load_graph()

    def load_graph(self):
        self.vectorstore = VectorStore()
        self.tools = [self.vectorstore.as_tool(), arxiv_search, search]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = MemorySaver()
        self._build_graph()

    def upload_document(self, file_path: str, filename):
        self.uploaded_docs.append(filename)
        docs = parse_pdf(file_path)
        self.vectorstore.add_documents(docs)
        return True

    def chat(self, state):
        response = self.llm_with_tools.invoke(state["messages"])
        state["messages"] += [response]
        return state

    def yt_node(self, state):
        state["yt_links"] = eval(yt_search.invoke(state["messages"][-1].content))
        return state

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("yt_node", self.yt_node)
        workflow.add_node("chat", self.chat)
        workflow.add_node("tools", ToolNode(tools=self.tools))
        workflow.add_conditional_edges("chat", tools_condition)
        workflow.add_edge("tools", "chat")
        workflow.add_edge("yt_node", "chat")
        workflow.add_edge(START, "yt_node")
        self.graph = workflow.compile(checkpointer=self.memory)

    def stream(self, query: str):
        response = self.graph.stream(
            {"messages": [{"role": "user", "content": query}]},
            self.config,
            stream_mode="messages",
        )
        return response

    def get_state(self):
        return self.graph.get_state(self.config)
