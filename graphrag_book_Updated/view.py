import networkx as nx
from matplotlib import pyplot as plt


def load_graph(graphml: str | nx.Graph) -> nx.Graph:
    """Load a graph from a graphml file or a networkx graph."""
    return nx.parse_graphml(graphml) if isinstance(graphml, str) else graphml

with open('/media/devuser/4cb8da84-6521-4e0b-9aeb-436f495aba56/GraphRag/graphrag_book_Updated/output/20240805-161813/artifacts/embedded_graph.2.graphml', 'r', encoding='utf-8') as f:
    graph_data = f.read()  # Read the GraphML data from the file
G = load_graph(graph_data)  # Use the read GraphML data to create a NetworkX Graph object
nx.draw(G, with_labels=True) # Load the graph
# G = load_graph('/media/devuser/4cb8da84-6521-4e0b-9aeb-436f495aba56/GraphRag/graphrag_book/output/20240805-153652/artifacts/clustered_graph.graphml')

# Draw the graph
# nx.draw(G, with_labels=True)  # Optional: Show node labels
plt.show()


#  /media/devuser/4cb8da84-6521-4e0b-9aeb-436f495aba56/GraphRag/graphrag_book/graphrag/index/utils/load_graph.py