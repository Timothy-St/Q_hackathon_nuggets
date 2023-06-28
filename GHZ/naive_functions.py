import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import IBMQ, QuantumCircuit
from networkx.algorithms.traversal.breadth_first_search import bfs_tree


def graph_to_tree(graph):
    """
    Convert a graph to a tree using BFS.
    :param graph: Input NetworkX graph.
    :return: A BFS tree.
    """
    # Create a BFS tree from the graph
    # You might want to change the source node (0 here) based on your needs
    tree = bfs_tree(graph, source=0)
    
    return tree



def generate_naive_noisy_qc_from_tree(T, multiplier):

    #Check that multiplier is odd
    if multiplier % 2 == 0:
        raise ValueError("Multiplier must be an odd number.")
    
    # Create a mapping from node labels to indices
    node_to_index = {node: index for index, node in enumerate(T.nodes)}

    # Create a quantum circuit
    qc = QuantumCircuit(len(T.nodes), len(T.nodes))

    # Apply Hadamard gate on the optimal root
    qc.h(node_to_index[0])

    # Create a list to keep track of processed nodes
    processed_nodes = [0]

    # Iterate until all nearest neighbors are processed
    while len(processed_nodes) < T.number_of_nodes():
        # Iterate over the processed nodes
        for node in processed_nodes:
            # Iterate over the nearest neighbors of the current processed node
            for neighbor in T.neighbors(node):
                # If the neighbor is not already processed, apply CX gate
                if neighbor not in processed_nodes:
                    # Repeat the CX gate multiplier times
                    for _ in range(multiplier):
                        qc.cx(node_to_index[node], node_to_index[neighbor])
                        #qc.barrier()
                    processed_nodes.append(neighbor)
    qc.measure(range(len(T.nodes)), range(len(T.nodes)))

    return qc