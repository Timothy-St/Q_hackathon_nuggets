import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import IBMQ, QuantumCircuit



def get_graph_from_backend(backend, fake=True, weight_function=lambda x: 1/(1-x)):

    if not fake:
        assert type(backend) == str, "backend must be a string"
        IBMQ.load_account()        
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = provider.get_backend(backend)  # something like 'ibmq_lima'      
        
    if fake:
        assert type(backend) == backend.__class__, "backend must be a backend object"

    try:
        config = backend.configuration()
        properties = backend.properties()
        coupling_map = config.coupling_map
    except:
        print("WARNING: most probable error is that you are using a V2 fake backend, you should use a V1 fake backend")

    G = nx.Graph()
    G.add_nodes_from(range(config.n_qubits))

    num_weight_one_edges = 0
    for qubit1, qubit2 in coupling_map:
        gate_error = properties.gate_error('cx', [qubit1, qubit2])
        print("gate error:\n", gate_error)

        # Skip adding edges with gate error of 1        
        if gate_error != 1:
            f_gate_error = weight_function(gate_error)
            G.add_edge(qubit1, qubit2, weight=f_gate_error)
        if gate_error == 1:
            num_weight_one_edges += 1

    # Keep only the largest connected component
    connected_components = list(nx.connected_components(G))

    num_connected_components = len(connected_components)

    # Calculate number of nodes in each component here before modifying the graph
    number_of_nodes_in_components = [len(component) for component in connected_components]

    
    if num_connected_components > 1:
        # sort by length, from largest to smallest
        connected_components.sort(key=len, reverse=True)
        largest_component = connected_components[0]

        # Keep only the largest component
        G = G.subgraph(largest_component).copy()
    
    print("REPORT: get_graph_from_backend:")
    print("Number of qubits: ", config.n_qubits)
    print("Number of weight 1 edges: ", num_weight_one_edges)
    print("Number of connected components: ", num_connected_components)
    print("with number of nodes in each component: ", number_of_nodes_in_components)
    print("Number of edges in new graph: ", G.number_of_edges())
    print("Number of nodes in new graph: ", G.number_of_nodes())

    return G




def get_tree_graph(G, root, weight_str='weight'):
    """
    returns a tree graph for a given root node
    """

    T = nx.Graph()
    T.add_nodes_from(G.nodes)

    shortest_paths = {node: nx.dijkstra_path(G, root, node, weight='weight') for node in G.nodes if node != root}

    for node, path in shortest_paths.items():
        if len(path) > 1:
            # Retrieve the weight from the original graph
            weight = G[path[-1]][path[-2]]['weight']
            # Add the edge to the tree
            T.add_edge(path[-1], path[-2], weight=weight)

    return T


def plot_graph(G):
    """
    Plots a given graph
    """
   # Draw the graph with edge labels
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    for key in labels:
        labels[key] = round(labels[key], 3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def minimal_max_distance_root(G):
    """
    Returns the root node that minimizes the maximum distance
    between the root and any other node, and the number of nodes on that path
    """

    # Initialize variables to track the optimal root
    min_max_distance = np.inf
    optimal_root = None
    optimal_path = []

    # Loop over all nodes
    for node in G.nodes:
        # Compute the shortest path distances from this node to all others
        paths = nx.single_source_dijkstra_path(G, node, weight='weight')

        # Find the longest distance
        max_distance_path = max(paths.values(), key=len)
        max_distance = len(max_distance_path) - 1

        # Update the optimal root if this node is better
        if max_distance < min_max_distance:
            min_max_distance = max_distance
            optimal_root = node
            optimal_path_length = len(max_distance_path)

    return optimal_root, min_max_distance, optimal_path_length




def generate_directed_tree_from_root(shortest_path_tree, root):
    
    # Create a directed graph
    directed_tree = nx.DiGraph()

    # Add the nodes from the shortest path tree
    directed_tree.add_nodes_from(shortest_path_tree.nodes)

    # Create a list to keep track of processed nodes
    processed_nodes = [root]

    # Iterate until all nearest neighbors are processed
    while len(processed_nodes) < shortest_path_tree.number_of_nodes():
        # Iterate over the processed nodes
        for node in processed_nodes:
            # Iterate over the nearest neighbors of the current processed node
            for neighbor in shortest_path_tree.neighbors(node):
                # If the neighbor is not already processed, add a directed edge
                if neighbor not in processed_nodes:
                    directed_tree.add_edge(node, neighbor)
                    processed_nodes.append(neighbor)

    return directed_tree


def order_neighbors_by_distance(graph, node):
    """
    Order the neighbors of a node in a graph based on the distance to the furthest reachable node through each neighbor.

    Parameters:
    - graph: A networkx.Graph or networkx.DiGraph object.
    - node: The node whose neighbors will be ordered.

    Returns:
    - A list of tuples, each containing a neighbor and the distance to the furthest node reachable through that neighbor,
      ordered by the distance in descending order.
    """
    neighbors = list(graph.neighbors(node))
    distances = []

    for neighbor in neighbors:
        lengths = nx.single_source_dijkstra_path_length(graph, neighbor)
        max_length = max(lengths.values())
        distances.append((neighbor, max_length))

    # Sort the neighbors based on the distances in descending order
    distances.sort(key=lambda x: x[1], reverse=True)

    return distances

def generate_qc_from_tree(shortest_path_tree, optimal_root):
    # Create a mapping from node labels to indices
    node_to_index = {node: index for index, node in enumerate(shortest_path_tree.nodes)}

    # Create a quantum circuit
    qc = QuantumCircuit(len(shortest_path_tree.nodes), len(shortest_path_tree.nodes))

    # Apply Hadamard gate on the optimal root
    qc.h(node_to_index[optimal_root])

    # Create a list to keep track of processed nodes
    processed_nodes = [optimal_root]

    # Iterate until all nearest neighbors are processed
    while len(processed_nodes) < shortest_path_tree.number_of_nodes():
        # Iterate over the processed nodes
        for node in processed_nodes:
            # Iterate over the nearest neighbors of the current processed node
            neighbors_distances = order_neighbors_by_distance(shortest_path_tree, node)
            ordered_neighbors = [neighbor for neighbor, distance in neighbors_distances]
            for neighbor in ordered_neighbors:
                # If the neighbor is not already processed, apply CX gate
                if neighbor not in processed_nodes:
                    qc.cx(node_to_index[node], node_to_index[neighbor])
                    processed_nodes.append(neighbor)
    qc.measure(range(len(shortest_path_tree.nodes)), range(len(shortest_path_tree.nodes)))

    return qc


def generate_noisy_qc_from_tree(shortest_path_tree, optimal_root, multiplier):
    # Check that multiplier is odd
    if multiplier % 2 == 0:
        raise ValueError("Multiplier must be an odd number.")
    
    # Create a mapping from node labels to indices
    node_to_index = {node: index for index, node in enumerate(shortest_path_tree.nodes)}

    # Create a quantum circuit
    qc = QuantumCircuit(len(shortest_path_tree.nodes), len(shortest_path_tree.nodes))

    # Apply Hadamard gate on the optimal root
    qc.h(node_to_index[optimal_root])

    # Create a list to keep track of processed nodes
    processed_nodes = [optimal_root]

    # Iterate until all nearest neighbors are processed
    while len(processed_nodes) < shortest_path_tree.number_of_nodes():
        # Iterate over the processed nodes
        for node in processed_nodes:
            # Iterate over the nearest neighbors of the current processed node
            neighbors_distances = order_neighbors_by_distance(shortest_path_tree, node)
            ordered_neighbors = [neighbor for neighbor, distance in neighbors_distances]
            for neighbor in ordered_neighbors:
                # If the neighbor is not already processed, apply CX gate
                if neighbor not in processed_nodes:
                    # Repeat the CX gate multiplier times
                    for _ in range(multiplier):
                        qc.cx(node_to_index[node], node_to_index[neighbor])
                        #qc.barrier()
                    processed_nodes.append(neighbor)
    qc.measure(range(len(shortest_path_tree.nodes)), range(len(shortest_path_tree.nodes)))

    return qc





########### OLD FUNCTIONS BELOW #####################





def get_graph_from_backend_old(backend, fake=True, weight_function=lambda x: x):

    if not fake:
        assert type(backend) == str, "backend must be a string"
        IBMQ.load_account()        
        provider = IBMQ.get_provider(hub='ibm-q')
        backend = provider.get_backend(backend)  # something like 'ibmq_lima'     
        
    if fake:
        assert type(backend) == backend.__class__, "backend must be a backend object"

    config = backend.configuration()
    properties = backend.properties()
    coupling_map = config.coupling_map

    G = nx.Graph()
    G.add_nodes_from(range(config.n_qubits))

    for qubit1, qubit2 in coupling_map:
        gate_error = properties.gate_error('cx', [qubit1, qubit2])
        print("gate errors:\n", gate_error)
        try:
            gate_error=weight_function(gate_error)
        except:
            ZeroDivisionError
            gate_error = 1e30
        G.add_edge(qubit1, qubit2, weight=gate_error)

    return G

def generate_qc_from_tree_old(shortest_path_tree, optimal_root):
    
    # Create a quantum circuit
    qc = QuantumCircuit(len(shortest_path_tree.nodes), len(shortest_path_tree.nodes))

    # Apply Hadamard gate on the optimal root
    qc.h(optimal_root)

    # Create a list to keep track of processed nodes
    processed_nodes = [optimal_root]

    # Iterate until all nearest neighbors are processed
    while len(processed_nodes) < shortest_path_tree.number_of_nodes():
        # Iterate over the processed nodes
        for node in processed_nodes:
            # Iterate over the nearest neighbors of the current processed node
            for neighbor in shortest_path_tree.neighbors(node):
                # If the neighbor is not already processed, apply CX gate
                if neighbor not in processed_nodes:
                    qc.cx(node, neighbor)
                    processed_nodes.append(neighbor)

    return qc


def generate_qc_from_tree_old_new(shortest_path_tree, optimal_root):
    
    # Create a quantum circuit
    qc = QuantumCircuit(len(shortest_path_tree.nodes))

    # Apply Hadamard gate on the optimal root
    qc.h(optimal_root)

    # Create a list to keep track of processed nodes
    processed_nodes = [optimal_root]

    # Iterate until all nearest neighbors are processed
    while len(processed_nodes) < shortest_path_tree.number_of_nodes():
        # Iterate over the processed nodes
        for node in processed_nodes:
            # Iterate over the nearest neighbors of the current processed node
            neighbors_distances = order_neighbors_by_distance(shortest_path_tree, node)
            ordered_neighbors = [neighbor for neighbor, distance in neighbors_distances]
            for neighbor in ordered_neighbors:
                # If the neighbor is not already processed, apply CX gate
                if neighbor not in processed_nodes:
                    qc.cx(node, neighbor)
                    processed_nodes.append(neighbor)

    return qc
