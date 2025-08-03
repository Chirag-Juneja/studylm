import networkx as nx
from studylm.core.llm import transformer


def build_knowledge_graph(documents):
    G = nx.Graph()
    for doc in documents:
        gdoc = transformer.process_response(doc)
        for node in gdoc.nodes:
            G.add_node(node.type, label=node.id)
        for edge in gdoc.relationships:
            G.add_edge(edge.source.id, edge.target.id, label=edge.type)
        yield G
