"""
Network analysis utilities for X Network Visualization.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx

from config import DEFAULT_CLOUTRANK_DAMPING, DEFAULT_CLOUTRANK_EPSILON, DEFAULT_CLOUTRANK_MAX_ITER
from utils.logging_config import get_logger
from data.network import NetworkData

logger = get_logger(__name__)

def compute_in_degree(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Computes the in-degree for each node in a directed graph.
    The in-degree is the number of incoming edges a node has.

    Args:
        graph: The networkx.DiGraph object representing the network.

    Returns:
        A dictionary mapping node IDs to their in-degree scores.
    """
    logger.info("Computing in-degree scores.")
    in_degrees = {node: float(degree) for node, degree in graph.in_degree()}
    return in_degrees

def compute_cloutrank(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Computes the CloutRank (PageRank) for each node in a directed graph.

    Args:
        graph: The networkx.DiGraph object representing the network.

    Returns:
        A dictionary mapping node IDs to their CloutRank scores.
    """
    logger.info("Computing CloutRank scores.")
    try:
        # Use networkx's built-in pagerank algorithm
        cloutrank = nx.pagerank(
            graph,
            alpha=DEFAULT_CLOUTRANK_DAMPING,
            max_iter=DEFAULT_CLOUTRANK_MAX_ITER,
            tol=DEFAULT_CLOUTRANK_EPSILON
        )
        return cloutrank
    except nx.NetworkXError as e:
        logger.error(f"Error computing CloutRank: {e}")
        return {node: 0.0 for node in graph.nodes()}


def select_top_nodes_for_visualization(
    network_data: NetworkData,
    importance_scores: Dict[str, float],
    max_accounts: int,
    community_labels: Optional[Dict[str, str]] = None,
    tweet_summaries: Optional[Dict[str, str]] = None,
) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
    """
    Selects the most important nodes and their details for a table view.

    Args:
        network_data: The NetworkData object representing the network.
        importance_scores: Dictionary of importance scores for each node.
        max_accounts: The maximum number of nodes to include in the table.
        community_labels: Optional dictionary mapping community IDs to names.
        tweet_summaries: Optional dictionary of tweet summaries for each node.

    Returns:
        A tuple containing:
        - A filtered nx.DiGraph object with only the selected nodes and their edges.
        - A list of dictionaries, where each dictionary represents a selected node with details
        for the table view.
    """
    logger.info(f"Selecting top {max_accounts} nodes for a table view.")
    
    graph = network_data.graph
    
    # Get the original node
    original_node_id = f"orig_{network_data.original_username}"
    
    # Sort nodes by importance score in descending order
    # Exclude the original node from this sorting
    sorted_nodes = sorted(
        [
            (node_id, score) 
            for node_id, score in importance_scores.items() 
            if node_id != original_node_id
        ], 
        key=lambda item: item[1], 
        reverse=True
    )

    # Safely convert max_accounts to an integer, handling the case where it's a dict
    try:
        if isinstance(max_accounts, dict):
            accounts_to_show = int(max_accounts.get('value', 0))
        else:
            accounts_to_show = int(max_accounts)
    except (ValueError, TypeError):
        logger.error(f"Could not convert max_accounts ({max_accounts}) to an integer. Using default 50.")
        accounts_to_show = 50

    # Get the top N node IDs
    top_node_ids = {node_id for node_id, score in sorted_nodes[:accounts_to_show]}
    top_node_ids.add(original_node_id) # Always include the original node
    
    # Create a subgraph with the selected nodes
    filtered_graph = graph.subgraph(top_node_ids)

    # Create the list of dictionaries for the table
    table_data = []
    for node_id in top_node_ids:
        if node_id in graph.nodes:
            node_data = graph.nodes[node_id].copy()
            
            # Get the importance score
            importance = importance_scores.get(node_id, 0.0)

            # Create a selection reason
            selection_reason = f"Selected due to a high importance score of {importance:.4f}"
            
            # Add community info if available
            if community_labels and 'community_id' in node_data:
                community_id = node_data['community_id']
                community_name = community_labels.get(community_id, f"Community {community_id}")
                node_data['community_name'] = community_name
                selection_reason += f" and belonging to the '{community_name}' community."
            else:
                selection_reason += "."
            
            table_data.append({
                'username': node_id,
                'importance_score': importance,
                'selection_reason': selection_reason,
                'tweet_summary': tweet_summaries.get(node_id, "No summary available.") if tweet_summaries else "No summary available."
            })
        
    return filtered_graph, table_data
