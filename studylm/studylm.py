import re

import uuid
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from studylm.utils.logger import get_logger
from studylm.config.constants import MODEL
from studylm.core.pdf_parser import parse_pdf
from studylm.tools.rag import VectorStore

logger = get_logger(__name__)


def remove_think_block(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class StudyLM:
    def __init__(self):
        self.llm = init_chat_model(model=MODEL, model_provider="ollama")
        self.config = {"configurable": {"thread_id": "1"}}
        self.uploaded_docs = []
        self.load_graph()

    def load_graph(self):
        self.vectorstore = VectorStore()
        self.tools = [self.vectorstore.as_tool()]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = MemorySaver()
        self._build_graph()

    def upload_document(self, file_path: str, filename):
        self.uploaded_docs.append(filename)
        docs = parse_pdf(file_path)
        self.vectorstore.add_documents(docs)
        return True

    def chat(self, state: MessagesState):
        response = (self.llm_with_tools.invoke(state["messages"]))
        return {"messages": state["messages"]+[response]}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("chat", self.chat)
        workflow.add_node("tools", ToolNode(tools=self.tools))
        workflow.add_conditional_edges("chat", tools_condition)
        workflow.add_edge("tools", "chat")
        workflow.add_edge(START, "chat")
        self.graph = workflow.compile(checkpointer=self.memory)

    def stream(self, query: str):
        response = self.graph.stream(
            {"messages": [{"role": "user", "content": query}]},
            self.config,
            stream_mode="messages"
        )
        return response

    def get_state(self):
        return self.graph.get_state(self.config)
