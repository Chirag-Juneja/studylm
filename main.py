from pyvis.network import Network
from studylm.utils.logger import get_logger
from studylm.core.knowledge_graph import build_knowledge_graph
from studylm.core.pdf_parser import parse_pdf
from tqdm import tqdm

logger = get_logger(__name__)

path = "./data/book.pdf"
chunks = parse_pdf(path)

for G in tqdm(build_knowledge_graph(chunks), total=len(chunks)):
    net = Network()
    net.from_nx(G)
    net.save_graph("artifacts/graph.html")
