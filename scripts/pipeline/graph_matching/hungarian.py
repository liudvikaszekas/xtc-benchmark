"""
Consolidated graph matching module combining Hungarian algorithm and semantic matching.

This module provides:
1. Graph data structures (Node, Edge, Graph)
2. Graph loading and embedding computation
3. Hungarian algorithm for graph edit distance
4. Semantic graph matching with NetworkX
5. Helper functions for graph conversion and visualization
"""

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer


 


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class Node:
    id: str
    label: str
    embedding: Optional[np.ndarray]
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    embedding: Optional[np.ndarray]


@dataclass
class Graph:
    nodes: Dict[str, Node]
    edges: List[Edge]
    # adjacency list: node_id -> list of edge embeddings of incident edges
    incident_edge_embeddings: Dict[str, List[np.ndarray]]
    # adjacency structure: node_id -> set of neighbor node_ids
    neighbors: Dict[str, Set[str]]
    # edge lookup: (source, target) -> Edge
    edge_map: Dict[Tuple[str, str], Edge]


# ============================================================================
# Graph Loading and Embedding
# ============================================================================

def load_graph_from_data(data: Dict, source_name: str = "data") -> Graph:
    """Load a graph from a dictionary.
    
    Expected format:
    {
        "nodes": [{"id": "...", "label": "..."}, ...],
        "edges": [{"source": "...", "target": "...", "relation": "..."}, ...]
    }
    """
    nodes_raw = data.get("nodes", [])
    edges_raw = data.get("edges", [])

    # Build node objects with label and attributes. Embeddings are computed later.
    nodes = {}
    for n in nodes_raw:
        attrs = n.get("attributes", {})
        if attrs is None:
            attrs = {}
        nodes[n["id"]] = Node(
            id=n["id"], 
            label=n.get("label", n["id"]), 
            embedding=None,
            attributes=attrs
        )

    # Build edges with relation text
    edges: List[Edge] = []
    edge_map: Dict[Tuple[str, str], Edge] = {}
    neighbors: Dict[str, Set[str]] = {nid: set() for nid in nodes.keys()}
    
    for e in edges_raw:
        rel_text = e.get("relation")
        if rel_text is None:
            raise ValueError(
                f"Edge from {e.get('source')} to {e.get('target')} is missing a 'relation' "
                f"field in {source_name}. Please add a relation label (e.g. 'is', 'has', 'related to')."
            )
        
        edge = Edge(source=e["source"], target=e["target"], relation=rel_text, embedding=None)
        edges.append(edge)
        edge_map[(e["source"], e["target"])] = edge
        
        # Build adjacency (treating as undirected for neighbor tracking)
        if e["source"] in neighbors:
            neighbors[e["source"]].add(e["target"])
        if e["target"] in neighbors:
            neighbors[e["target"]].add(e["source"])
    
    incident: Dict[str, List[np.ndarray]] = {nid: [] for nid in nodes.keys()}
    
    return Graph(
        nodes=nodes, 
        edges=edges, 
        incident_edge_embeddings=incident, 
        neighbors=neighbors, 
        edge_map=edge_map
    )


def load_graph(path: str) -> Graph:
    """Load a graph from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return load_graph_from_data(raw, source_name=path)


def compute_text_embeddings_for_graph(graph: Graph, model: SentenceTransformer) -> None:
    """Compute embeddings for node labels and edge relation texts.
    
    This function mutates the passed Graph in-place.
    - Node embedding is computed from its label string.
    - Edge embedding combines: "source_label relation target_label"
      This ensures edges are structurally aware.
    """
    # Node embeddings
    node_ids = list(graph.nodes.keys())
    texts = [graph.nodes[nid].label for nid in node_ids]
    if len(texts) > 0:
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        for nid, emb in zip(node_ids, embs):
            graph.nodes[nid].embedding = np.array(emb, dtype=np.float32)

    # Edge embeddings: combine relation with endpoint node labels for structural awareness
    edge_texts: List[str] = []
    for e in graph.edges:
        source_label = graph.nodes[e.source].label
        target_label = graph.nodes[e.target].label
        edge_text = f"{source_label} {e.relation} {target_label}"
        edge_texts.append(edge_text)

    if len(edge_texts) > 0:
        eembs = model.encode(edge_texts, convert_to_numpy=True, normalize_embeddings=True)
        for e, emb in zip(graph.edges, eembs):
            e.embedding = np.array(emb, dtype=np.float32)

    # Build incident edge embeddings list
    incident: Dict[str, List[np.ndarray]] = {nid: [] for nid in graph.nodes.keys()}
    for e in graph.edges:
        if e.embedding is None:
            raise ValueError(
                f"Edge embedding for relation '{e.relation}' on edge "
                f"{e.source}->{e.target} was not computed"
            )
        if e.source in incident:
            incident[e.source].append(e.embedding)
        if e.target in incident:
            incident[e.target].append(e.embedding)
    graph.incident_edge_embeddings = incident


# ============================================================================
# Similarity Functions
# ============================================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Handles None inputs and zero vectors gracefully.
    """
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def edge_set_similarity(Ei: List[np.ndarray], Ej: List[np.ndarray]) -> float:
    """Fuzzy Jaccard similarity between two sets of edge embeddings.
    
    Algorithm (proper fuzzy Jaccard):
      - Compute pairwise cosine similarities between edges in Ei and Ej (clamped to [0,1])
      - Perform greedy one-to-one matching: pick largest remaining similarity,
        add to intersection mass, mark row and column as used
      - intersection = sum(matched similarities)
      - union = |Ei| + |Ej| - intersection (cardinality-based union)
      - Return intersection / union (or 0.0 if union == 0)
    """
    # Fast exits
    if len(Ei) == 0 and len(Ej) == 0:
        return 1.0
    if len(Ei) == 0 or len(Ej) == 0:
        return 0.0

    Sij = np.zeros((len(Ei), len(Ej)), dtype=np.float32)
    for ii, ei in enumerate(Ei):
        for jj, ej in enumerate(Ej):
            # cosine similarity may be negative; clamp to 0..1
            Sij[ii, jj] = max(0.0, _cosine_similarity(ei, ej))

    # Greedy one-to-one matching to compute intersection mass
    intersection = 0.0
    used_rows = set()
    used_cols = set()
    # Flatten indices sorted by similarity descending
    flat_idx = np.argsort(-Sij, axis=None)
    ii_idx, jj_idx = np.unravel_index(flat_idx, Sij.shape)
    for r, c in zip(ii_idx, jj_idx):
        if r in used_rows or c in used_cols:
            continue
        sim = float(Sij[r, c])
        if sim <= 0.0:
            break
        intersection += sim
        used_rows.add(r)
        used_cols.add(c)

    # Union based on cardinality: total edges minus intersection
    union = float(len(Ei) + len(Ej)) - intersection
    if union <= 0.0:
        return 1.0  # both empty after accounting for perfect matches
    return float(max(0.0, min(1.0, intersection / union)))


def structural_similarity(neighbors_i: Set[str], neighbors_j: Set[str]) -> float:
    """Compute structural similarity based on neighborhood overlap (Jaccard similarity).
    
    Returns value in [0, 1]. If both nodes have no neighbors, returns 1.0 (both isolated).
    """
    if len(neighbors_i) == 0 and len(neighbors_j) == 0:
        return 1.0
    if len(neighbors_i) == 0 or len(neighbors_j) == 0:
        return 0.0
    intersection = len(neighbors_i & neighbors_j)
    union = len(neighbors_i | neighbors_j)
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def attribute_similarity(attrs1: Dict[str, Any], attrs2: Dict[str, Any], model: Optional[SentenceTransformer] = None) -> float:
    """Compute similarity between two attribute dictionaries using BERT embeddings.
    
    Attributes are sorted by key, then pairwise scores are computed for each attribute type
    using BERT embeddings and normalized across all attributes together.
    
    IMPORTANT: If an attribute exists in one node but not the other, it is penalized
    (treated as 0 similarity for that attribute).
    
    Args:
        attrs1: First attribute dictionary (e.g., {"color": ["red", "blue"], "size": ["large"]})
        attrs2: Second attribute dictionary
        model: SentenceTransformer model for computing embeddings (same as used for label similarity)
    
    Returns:
        Similarity score in [0, 1]
    """
    if not attrs1 and not attrs2:
        return 1.0
    
    if model is None:
        raise ValueError("model parameter is required for attribute_similarity")
    
    # Sort attribute keys for consistent comparison
    keys1 = sorted(attrs1.keys()) if attrs1 else []
    keys2 = sorted(attrs2.keys()) if attrs2 else []
    
    # Get all unique attribute keys (union of both) - EXCLUDE visual_reasoning as it is meta-text
    all_keys = sorted([k for k in set(keys1) | set(keys2) if k != 'visual_reasoning'])
    
    # Compute pairwise similarities for each attribute type
    similarities = []
    for key in all_keys:
        # Get attribute values, defaulting to empty list if key doesn't exist
        # This ensures missing attributes are penalized
        val1 = attrs1.get(key, []) if attrs1 else []
        val2 = attrs2.get(key, []) if attrs2 else []
        
        # Ensure values are lists
        if not isinstance(val1, list):
            val1 = [val1] if val1 else []
        if not isinstance(val2, list):
            val2 = [val2] if val2 else []
        
        # Sort values for consistent comparison and join as string
        val1_sorted = sorted([str(v).lower() for v in val1])
        val2_sorted = sorted([str(v).lower() for v in val2])
        
        # Join sorted values as a string
        str1 = " ".join(val1_sorted)
        str2 = " ".join(val2_sorted)
        
        if not str1 and not str2:
            # Both empty for this attribute (attribute missing in both)
            sim = 1.0
        elif not str1 or not str2:
            # One has the attribute, the other doesn't - PENALIZE (0 similarity)
            sim = 0.0
        else:
            # Both have the attribute, compute BERT embedding similarity
            emb1 = model.encode([str1], convert_to_numpy=True, normalize_embeddings=True)[0]
            emb2 = model.encode([str2], convert_to_numpy=True, normalize_embeddings=True)[0]
            sim = _cosine_similarity(emb1, emb2)
        
        similarities.append(sim)
    
    # Normalize: average across all attributes
    if similarities:
        return float(np.mean(similarities))
    return 0.0


def compute_edge_similarity_with_attributes(
    edge_emb1: Optional[np.ndarray],
    edge_emb2: Optional[np.ndarray],
    u1_attrs: Dict[str, Any],
    u2_attrs: Dict[str, Any],
    v1_attrs: Dict[str, Any],
    v2_attrs: Dict[str, Any],
    model: Optional[SentenceTransformer] = None
) -> float:
    """Compute edge similarity combining embedding and node attributes.
    
    Args:
        edge_emb1: Edge embedding for first edge
        edge_emb2: Edge embedding for second edge
        u1_attrs: Attributes of source node in first edge
        u2_attrs: Attributes of source node in second edge
        v1_attrs: Attributes of target node in first edge
        v2_attrs: Attributes of target node in second edge
        model: SentenceTransformer model for computing attribute embeddings
    
    Returns:
        Combined edge similarity: 40% edge embedding + 60% node attributes
    """
    # Normalize attributes (handle None)
    if u1_attrs is None:
        u1_attrs = {}
    if u2_attrs is None:
        u2_attrs = {}
    if v1_attrs is None:
        v1_attrs = {}
    if v2_attrs is None:
        v2_attrs = {}
    
    # Edge embedding similarity (includes node labels)
    sim_edge_emb = _cosine_similarity(edge_emb1, edge_emb2)
    
    # Node attribute similarity for source and target nodes
    sim_u_attr = attribute_similarity(u1_attrs, u2_attrs, model)
    sim_v_attr = attribute_similarity(v1_attrs, v2_attrs, model)
    sim_attr_combined = (sim_u_attr + sim_v_attr) / 2.0
    
    # Combined edge similarity: 40% edge embedding + 60% node attributes
    sim_edge = 0.4 * sim_edge_emb + 0.6 * sim_attr_combined
    return sim_edge


# ============================================================================
# Hungarian Algorithm for Graph Edit Distance
# ============================================================================

def build_cost_matrix(
    G1: Graph,
    G2: Graph,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build substitution cost matrix for Hungarian algorithm.
    
    Cost formula:
      C[i,j] = alpha * (1 - node_sim) + beta * (1 - edge_sim)
    
    Where:
      - node_sim = cosine similarity between node label embeddings
      - edge_sim = fuzzy Jaccard similarity between incident edge label embeddings
    
    Args:
      alpha: Weight for node label dissimilarity
      beta: Weight for edge label dissimilarity
    
    Lower cost = more similar. alpha + beta should sum to 1.0 for normalized costs.
    Returns the cost matrix and the ordered node id lists (rows=G1_ids, cols=G2_ids).
    """
    ids1 = list(G1.nodes.keys())
    ids2 = list(G2.nodes.keys())
    n1, n2 = len(ids1), len(ids2)
    C = np.zeros((n1, n2), dtype=np.float32)
    
    for i, id1 in enumerate(ids1):
        n1e = G1.nodes[id1].embedding
        E1 = G1.incident_edge_embeddings.get(id1, [])
        
        for j, id2 in enumerate(ids2):
            n2e = G2.nodes[id2].embedding
            E2 = G2.incident_edge_embeddings.get(id2, [])
            
            # Node label similarity
            if n1e is None or n2e is None:
                node_sim = 0.0
            else:
                node_sim = _cosine_similarity(n1e, n2e)
            
            # Edge label similarity (fuzzy Jaccard)
            edge_sim = edge_set_similarity(E1, E2)
            
            # Combined cost
            node_dist = 1.0 - node_sim
            edge_dist = 1.0 - edge_sim
            C[i, j] = alpha * node_dist + beta * edge_dist
            
    return C, ids1, ids2


def pad_to_square(C: np.ndarray, penalty: float) -> Tuple[np.ndarray, int, int]:
    """Pad a rectangular cost matrix to square by adding dummy rows/cols."""
    n_rows, n_cols = C.shape
    if n_rows == n_cols:
        return C, n_rows, n_cols
    n = max(n_rows, n_cols)
    C_square = np.full((n, n), penalty, dtype=C.dtype)
    C_square[:n_rows, :n_cols] = C
    return C_square, n_rows, n_cols


def run_hungarian(
    C: np.ndarray, 
    ids1: List[str], 
    ids2: List[str], 
    node_del_cost: float,
    node_ins_cost: float,
    match_threshold: float
) -> Tuple[List[Tuple[str, str, float]], List[str], List[str]]:
    """Execute Hungarian assignment with explicit deletion/insertion costs.
    
    Args:
      C: Cost matrix for substitutions (matching nodes)
      ids1, ids2: Node IDs for each graph
      node_del_cost: Cost to delete a node from G1 (leave unmatched)
      node_ins_cost: Cost to insert a node into G1 (leave G2 node unmatched)
      match_threshold: Maximum substitution cost to accept a match
    
    Returns:
      - matched: List of (id1, id2, substitution_cost) tuples
      - unmatched_1: Node IDs from G1 that were deleted
      - unmatched_2: Node IDs from G2 that were inserted
    """
    n_rows, n_cols = C.shape
    
    if n_rows == n_cols:
        C_square = C.copy()
    else:
        n = max(n_rows, n_cols)
        C_square = np.zeros((n, n), dtype=C.dtype)
        C_square[:n_rows, :n_cols] = C
        # Pad rows (G1 nodes) with insertion cost for dummy G2 nodes
        if n_rows < n:
            C_square[n_rows:, :n_cols] = node_ins_cost
        # Pad cols (G2 nodes) with deletion cost for dummy G1 nodes
        if n_cols < n:
            C_square[:n_rows, n_cols:] = node_del_cost
        # Dummy-to-dummy should have zero cost
        if n_rows < n and n_cols < n:
            C_square[n_rows:, n_cols:] = 0.0
    
    row_ind, col_ind = linear_sum_assignment(C_square)

    matched: List[Tuple[str, str, float]] = []

    # Build lookup for assignments
    assign_for_row: Dict[int, int] = {r: c for r, c in zip(row_ind, col_ind)}
    assign_for_col: Dict[int, int] = {c: r for r, c in zip(row_ind, col_ind)}

    for r, c in zip(row_ind, col_ind):
        cost = float(C_square[r, c])
        is_real_row = r < n_rows
        is_real_col = c < n_cols
        if is_real_row and is_real_col and cost <= match_threshold:
            matched.append((ids1[r], ids2[c], cost))

    # Unmatched in G1 (deletions): assigned to dummy or assigned with too-high cost
    unmatched_1: List[str] = []
    for r in range(n_rows):
        c = assign_for_row.get(r, None)
        if c is None:
            unmatched_1.append(ids1[r])
            continue
        cost = float(C_square[r, c])
        if c >= n_cols or cost > match_threshold:
            unmatched_1.append(ids1[r])

    # Unmatched in G2 (insertions): assigned to dummy or assigned with too-high cost
    unmatched_2: List[str] = []
    for c in range(n_cols):
        r = assign_for_col.get(c, None)
        if r is None:
            unmatched_2.append(ids2[c])
            continue
        cost = float(C_square[r, c])
        if r >= n_rows or cost > match_threshold:
            unmatched_2.append(ids2[c])

    return matched, unmatched_1, unmatched_2


def compute_graph_edit_distance(
    G1: Graph,
    G2: Graph,
    matched: List[Tuple[str, str, float]],
    unmatched_1: List[str],
    unmatched_2: List[str],
    node_del_cost: float,
    node_ins_cost: float,
    edge_del_cost: float,
    edge_ins_cost: float,
) -> Dict[str, Any]:
    """Compute the total graph edit distance given node matching results.
    
    Returns a dictionary with:
      - total_cost: Total GED
      - node_costs: Dict with substitution, deletion, insertion costs
      - edge_costs: Dict with deletion, insertion costs
      - breakdown: Detailed cost breakdown
    """
    # Node costs
    node_sub_cost_total = sum(cost for _, _, cost in matched)
    node_del_cost_total = len(unmatched_1) * node_del_cost
    node_ins_cost_total = len(unmatched_2) * node_ins_cost
    
    # Build set of matched node pairs
    matched_map = {id1: id2 for id1, id2, _ in matched}
    matched_set_1 = set(matched_map.keys())
    matched_set_2 = set(matched_map.values())
    
    # Count edge operations
    edges_deleted = 0
    edges_inserted = 0
    
    # Edges in G1: deleted if either endpoint is unmatched or no corresponding edge in G2
    for edge in G1.edges:
        if edge.source not in matched_set_1 or edge.target not in matched_set_1:
            edges_deleted += 1
        else:
            # Both endpoints matched, check if corresponding edge exists in G2
            mapped_source = matched_map[edge.source]
            mapped_target = matched_map[edge.target]
            # Check both directions for undirected graph
            if ((mapped_source, mapped_target) not in G2.edge_map and
                (mapped_target, mapped_source) not in G2.edge_map):
                edges_deleted += 1
    
    # Edges in G2: inserted if either endpoint is unmatched or no corresponding edge in G1
    for edge in G2.edges:
        if edge.source not in matched_set_2 or edge.target not in matched_set_2:
            edges_inserted += 1
        else:
            # Both endpoints matched, check if corresponding edge exists in G1
            rev_map = {v: k for k, v in matched_map.items()}
            mapped_source = rev_map[edge.source]
            mapped_target = rev_map[edge.target]
            if ((mapped_source, mapped_target) not in G1.edge_map and
                (mapped_target, mapped_source) not in G1.edge_map):
                edges_inserted += 1
    
    edge_del_cost_total = edges_deleted * edge_del_cost
    edge_ins_cost_total = edges_inserted * edge_ins_cost
    
    total_cost = (node_sub_cost_total + node_del_cost_total + node_ins_cost_total +
                  edge_del_cost_total + edge_ins_cost_total)
    
    return {
        "total_cost": total_cost,
        "node_costs": {
            "substitution": node_sub_cost_total,
            "deletion": node_del_cost_total,
            "insertion": node_ins_cost_total,
        },
        "edge_costs": {
            "deletion": edge_del_cost_total,
            "insertion": edge_ins_cost_total,
        },
        "breakdown": {
            "nodes_matched": len(matched),
            "nodes_deleted": len(unmatched_1),
            "nodes_inserted": len(unmatched_2),
            "edges_deleted": edges_deleted,
            "edges_inserted": edges_inserted,
        }
    }


# ============================================================================
# NetworkX Integration and Semantic Matching
# ============================================================================

def convert_hungarian_graph_to_nx(hg: Graph) -> nx.MultiDiGraph:
    """Convert the Graph dataclass to a networkx.MultiDiGraph.
    
    Assumes edge and node embeddings may already be present 
    (after compute_text_embeddings_for_graph).
    """
    G = nx.MultiDiGraph()
    for nid, node in hg.nodes.items():
        attrs = {"label": node.label}
        if getattr(node, "embedding", None) is not None:
            attrs["embedding"] = node.embedding
        if getattr(node, "attributes", None) is not None:
            attrs["attributes"] = node.attributes
        G.add_node(nid, **attrs)
    for e in hg.edges:
        attrs = {"label": e.relation}
        if getattr(e, "embedding", None) is not None:
            attrs["embedding"] = e.embedding
        G.add_edge(e.source, e.target, **attrs)
    return G


def attach_embeddings_to_nx(G: nx.MultiDiGraph, model: SentenceTransformer):
    """Compute and attach node and edge embeddings for a networkx graph.
    
    Node embedding: node label
    Edge embedding: "source_label relation target_label"
    """
    # Node embeddings
    node_ids = list(G.nodes())
    texts = [G.nodes[n].get("label", str(n)) for n in node_ids]
    if len(texts) > 0:
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        for nid, emb in zip(node_ids, embs):
            G.nodes[nid]["embedding"] = np.array(emb, dtype=np.float32)

    # Edge embeddings
    edge_list = list(G.edges(keys=True))
    edge_texts = []
    for u, v, k in edge_list:
        src_label = G.nodes[u].get("label", str(u))
        tgt_label = G.nodes[v].get("label", str(v))
        rel = G.edges[u, v, k].get("label", "")
        edge_texts.append(f"{src_label} {rel} {tgt_label}")
    if len(edge_texts) > 0:
        eembs = model.encode(edge_texts, convert_to_numpy=True, normalize_embeddings=True)
        for (u, v, k), emb in zip(edge_list, eembs):
            G.edges[u, v, k]["embedding"] = np.array(emb, dtype=np.float32)


def node_subst_cost(node1_attrs, node2_attrs):
    """Cost for substituting nodes using computed embeddings (for NetworkX).
    
    Returns 0.0 if semantic distance < 0.5, else 1.0.
    Raises error if embeddings are missing.
    """
    e1 = node1_attrs.get('embedding')
    e2 = node2_attrs.get('embedding')

    if isinstance(e1, np.ndarray) and isinstance(e2, np.ndarray):
        sim = _cosine_similarity(e1, e2)
        dist = 1.0 - sim
        return 0.0 if dist < 0.5 else 1.0

    raise ValueError("Node embeddings missing for substitution cost calculation.")


def edge_subst_cost(edge1_attrs, edge2_attrs):
    """Cost for substituting edges using computed embeddings (for NetworkX).
    
    Uses edge attribute 'embedding' (set by compute_text_embeddings_for_graph).
    If embeddings are present, cost = 1 - cosine_similarity.
    Falls back to label equality.
    """
    e1 = edge1_attrs.get('embedding')
    e2 = edge2_attrs.get('embedding')

    if isinstance(e1, np.ndarray) and isinstance(e2, np.ndarray):
        sim = _cosine_similarity(e1, e2)
        return float(max(0.0, 1.0 - sim))

    w1 = edge1_attrs.get('label', '')
    w2 = edge2_attrs.get('label', '')
    if not w1 or not w2:
        return 1.0
    return 0.0 if w1 == w2 else 1.0


def precompute_node_edge_lists(G: nx.MultiDiGraph) -> Dict[str, List[Tuple]]:
    """
    Pre-compute edge lists for all nodes to avoid repeated traversals.
    
    Returns:
        Dict mapping node_id -> list of (u, v, data) tuples for edges incident to that node
    """
    node_edges = {n: [] for n in G.nodes}
    
    for u, v, k, data in G.edges(keys=True, data=True):
        node_edges[u].append((u, v, data))
        node_edges[v].append((u, v, data))
    
    return node_edges


def compute_attribute_embeddings_per_key(attrs: Dict[str, Any], model: SentenceTransformer) -> Dict[str, np.ndarray]:
    """
    Pre-compute embeddings for each attribute key separately.
    
    This preserves the exact semantics of the original implementation:
    - Each attribute key is encoded separately
    - Values are sorted and joined as strings
    - Returns a dict mapping attribute_key -> embedding
    
    Args:
        attrs: Attribute dictionary (e.g., {"color": ["red", "blue"], "size": ["large"]})
        model: SentenceTransformer model
    
    Returns:
        Dict mapping attribute_key -> embedding vector
    """
    if not attrs:
        return {}
    
    attr_embeddings = {}
    
    for key in sorted(attrs.keys()):
        if key == 'visual_reasoning':
            continue
        val = attrs.get(key, [])
        
        # Ensure values are lists
        if not isinstance(val, list):
            val = [val] if val else []
        
        # Sort values for consistent comparison
        val_sorted = sorted([str(v).lower() for v in val])
        
        # Join sorted values as a string
        attr_text = " ".join(val_sorted)
        
        # Only encode if non-empty
        if attr_text:
            emb = model.encode([attr_text], convert_to_numpy=True, normalize_embeddings=True)[0]
            attr_embeddings[key] = np.array(emb, dtype=np.float32)
    
    return attr_embeddings


def precompute_all_attribute_embeddings(
    G: nx.MultiDiGraph, 
    model: SentenceTransformer
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Pre-compute attribute embeddings for all nodes in the graph.
    
    For each node, computes embeddings for each attribute key separately.
    This preserves exact semantics of the original implementation.
    
    Returns:
        Dict mapping node_id -> {attribute_key -> embedding}
    """
    attr_embeddings = {}
    
    for node_id in G.nodes:
        attrs = G.nodes[node_id].get('attributes', {})
        if attrs is None:
            attrs = {}
        attr_embeddings[node_id] = compute_attribute_embeddings_per_key(attrs, model)
    
    return attr_embeddings


def fast_attribute_similarity(
    attr_emb1: Dict[str, np.ndarray], 
    attr_emb2: Dict[str, np.ndarray]
) -> float:
    """
    Fast attribute similarity using pre-computed per-key embeddings.
    
    This exactly matches the original semantics:
    - Compare each attribute key independently
    - Penalize if one has an attribute and the other doesn't
    - Average similarities across all attribute keys
    
    Args:
        attr_emb1: Pre-computed attribute embeddings for first node {key -> embedding}
        attr_emb2: Pre-computed attribute embeddings for second node {key -> embedding}
    
    Returns:
        Similarity score in [0, 1]
    """
    # If both have no attributes, perfect match
    if not attr_emb1 and not attr_emb2:
        return 1.0
    
    # Get all unique attribute keys (union of both)
    # Sort all keys for consistent order - EXCLUDE visual_reasoning as it is meta-text
    all_keys = sorted([k for k in set(attr_emb1.keys()) | set(attr_emb2.keys()) if k != 'visual_reasoning'])
    
    if not all_keys:
        return 1.0
    
    # Compute pairwise similarities for each attribute type
    similarities = []
    for key in all_keys:
        emb1 = attr_emb1.get(key)
        emb2 = attr_emb2.get(key)
        
        if emb1 is None and emb2 is None:
            # Both empty for this attribute (shouldn't happen given all_keys construction)
            sim = 1.0
        elif emb1 is None or emb2 is None:
            # One has the attribute, the other doesn't - PENALIZE (exact original semantics)
            sim = 0.0
        else:
            # Both have the attribute, compute cosine similarity
            sim = max(0.0, _cosine_similarity(emb1, emb2))
        
        similarities.append(sim)
    
    # Average across all attributes (exact original semantics)
    return float(np.mean(similarities)) if similarities else 0.0


def compute_edge_similarity_with_precomputed_attrs(
    edge_emb1: Optional[np.ndarray],
    edge_emb2: Optional[np.ndarray],
    u1_attr_emb: Dict[str, np.ndarray],
    u2_attr_emb: Dict[str, np.ndarray],
    v1_attr_emb: Dict[str, np.ndarray],
    v2_attr_emb: Dict[str, np.ndarray]
) -> float:
    """
    Compute edge similarity using pre-computed per-key attribute embeddings.
    
    This is much faster than the original version because it uses pre-computed
    attribute embeddings rather than computing them on-the-fly.
    Now uses per-key embeddings to match exact original semantics.
    
    Args:
        edge_emb1: Edge embedding for first edge
        edge_emb2: Edge embedding for second edge
        u1_attr_emb: Pre-computed per-key attribute embeddings for source node in first edge
        u2_attr_emb: Pre-computed per-key attribute embeddings for source node in second edge
        v1_attr_emb: Pre-computed per-key attribute embeddings for target node in first edge
        v2_attr_emb: Pre-computed per-key attribute embeddings for target node in second edge
    
    Returns:
        Combined edge similarity: 40% edge embedding + 60% node attributes
    """
    # Edge embedding similarity (includes node labels)
    sim_edge_emb = _cosine_similarity(edge_emb1, edge_emb2)
    
    # Node attribute similarity using pre-computed per-key embeddings (exact original semantics)
    sim_u_attr = fast_attribute_similarity(u1_attr_emb, u2_attr_emb)
    sim_v_attr = fast_attribute_similarity(v1_attr_emb, v2_attr_emb)
    sim_attr_combined = (sim_u_attr + sim_v_attr) / 2.0
    
    # Combined edge similarity: 40% edge embedding + 60% node attributes
    sim_edge = 0.4 * sim_edge_emb + 0.6 * sim_attr_combined
    return sim_edge


def semantic_graph_matching(
    Ggt: nx.MultiDiGraph, 
    Gpred: nx.MultiDiGraph, 
    model: Optional[SentenceTransformer] = None, 
    verbose: bool = False, 
    node_sim_weight: float = 0.7, 
    edge_sim_weight: float = 0.3,
    gt_attr_embeddings: Optional[Dict] = None,
    pred_attr_embeddings: Optional[Dict] = None,
    gt_node_edges: Optional[Dict] = None,
    pred_node_edges: Optional[Dict] = None,
):
    """
    Optimized semantic graph matching using per-label Hungarian algorithm.
    """
    if verbose:
        print("\n=== PER-LABEL SEMANTIC GRAPH MATCHING ===")
    
    # Pre-compute edge lists for all nodes (Optimization)
    if gt_node_edges is None:
        gt_node_edges = precompute_node_edge_lists(Ggt)
    if pred_node_edges is None:
        pred_node_edges = precompute_node_edge_lists(Gpred)
    
    # Pre-compute attribute embeddings for all nodes (Optimization)
    if gt_attr_embeddings is None:
        if model is None:
            raise ValueError("model must be provided if gt_attr_embeddings is None")
        gt_attr_embeddings = precompute_all_attribute_embeddings(Ggt, model)
    if pred_attr_embeddings is None:
        if model is None:
            raise ValueError("model must be provided if pred_attr_embeddings is None")
        pred_attr_embeddings = precompute_all_attribute_embeddings(Gpred, model)
    
    # 1. Group nodes by label
    gt_nodes_by_label = defaultdict(list)
    for n in Ggt.nodes:
        lbl = Ggt.nodes[n].get('label', '').lower().strip()
        gt_nodes_by_label[lbl].append(n)
        
    pred_nodes_by_label = defaultdict(list)
    for n in Gpred.nodes:
        lbl = Gpred.nodes[n].get('label', '').lower().strip()
        pred_nodes_by_label[lbl].append(n)
        
    all_labels = set(gt_nodes_by_label.keys()) | set(pred_nodes_by_label.keys())
    
    matched_node_pairs = []
    node_mapping = {} # gt_id -> pred_id
    
    # Track unmatched for final stats
    all_gt_nodes = set(Ggt.nodes)
    all_pred_nodes = set(Gpred.nodes)
    matched_gt_nodes = set()
    matched_pred_nodes = set()

    # 2. Iterate per label
    if verbose:
        print(f"Processing {len(all_labels)} unique labels...")
    
    for lbl in all_labels:
        gt_group = gt_nodes_by_label.get(lbl, [])
        pred_group = pred_nodes_by_label.get(lbl, [])
        
        # If label only exists in one graph, all valid nodes are unmatched
        if not gt_group or not pred_group:
            continue
            
        # Build cost matrix for this specific label group
        n_gt = len(gt_group)
        n_pred = len(pred_group)
        cost_matrix = np.zeros((n_gt, n_pred))
        
        for i, n_gt_id in enumerate(gt_group):
            gt_attr_emb = gt_attr_embeddings[n_gt_id]
            gt_edges = gt_node_edges[n_gt_id]
            # gt_lbl is implied by group
            
            for j, n_pred_id in enumerate(pred_group):
                pred_attr_emb = pred_attr_embeddings[n_pred_id]
                pred_edges = pred_node_edges[n_pred_id]
                
                # A. Node Attribute Similarity
                sim_attr = fast_attribute_similarity(gt_attr_emb, pred_attr_emb)
                sim_node = sim_attr
                
                # B. Structural Edge Similarity
                edge_sim = 0.0
                if gt_edges and pred_edges:
                    # Edge matching sub-problem
                    # We want to see how well the edges structure matches
                    # We compare edges incident to these nodes
                    
                    edge_sub_cost = np.ones((len(gt_edges), len(pred_edges)))
                    for e_i, (_, _, data_gt) in enumerate(gt_edges):
                         u_gt, v_gt = gt_edges[e_i][0], gt_edges[e_i][1]
                         for e_j, (_, _, data_pred) in enumerate(pred_edges):
                             u_pred, v_pred = pred_edges[e_j][0], pred_edges[e_j][1]
                             
                             # Get attributes for all involved nodes to account for neighbor similarity
                             sim_val = compute_edge_similarity_with_precomputed_attrs(
                                data_gt.get('embedding'),
                                data_pred.get('embedding'),
                                gt_attr_embeddings[u_gt],
                                pred_attr_embeddings[u_pred],
                                gt_attr_embeddings[v_gt],
                                pred_attr_embeddings[v_pred]
                            )
                             edge_sub_cost[e_i, e_j] = 1.0 - sim_val
                    
                    # Hungarian for edges of this node pair
                    r_ind, c_ind = linear_sum_assignment(edge_sub_cost)
                    if len(r_ind) > 0:
                        # Average similarity of best matched edges
                        avg_cost = edge_sub_cost[r_ind, c_ind].mean()
                        edge_sim = 1.0 - avg_cost

                # C. Combined Cost
                # Weighted: default 70% attributes, 30% structure
                # If neither has edges, edge_sim is 0, which penalizes nodes with no edges slightly less?
                # Actually if both have no edges, we should rely 100% on attributes.
                if not gt_edges and not pred_edges:
                    combined_sim = sim_node
                elif not gt_edges or not pred_edges:
                    # One has edges, one doesn't -> mismatch in structure
                    # Penalty: structural score is 0
                    combined_sim = node_sim_weight * sim_node + edge_sim_weight * 0.0
                else:
                    combined_sim = node_sim_weight * sim_node + edge_sim_weight * edge_sim
                    
                cost_matrix[i, j] = 1.0 - combined_sim

        # Run Hungarian for this group
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
             cost = cost_matrix[r, c]
             # Accept best match since they belong to the same label group
             gt_id = gt_group[r]
             pred_id = pred_group[c]
             
             node_mapping[gt_id] = pred_id
             matched_gt_nodes.add(gt_id)
             matched_pred_nodes.add(pred_id)
             
             matched_node_pairs.append((
                pred_id, gt_id, 
                Gpred.nodes[pred_id].get('label', ''), 
                Ggt.nodes[gt_id].get('label', '')
            ))

    # Compile unmatched lists
    unmatched_gt_nodes_list = [(n, Ggt.nodes[n].get('label', '')) for n in all_gt_nodes if n not in matched_gt_nodes]
    unmatched_pred_nodes_list = [(n, Gpred.nodes[n].get('label', '')) for n in all_pred_nodes if n not in matched_pred_nodes]
    
    if verbose:
        print(f"Matched {len(matched_node_pairs)} node pairs")
    
    # --- Stage 2: Edge Matching ---
    if verbose:
        print("Stage 2: Edge matching...")
    matched_edge_label_pairs = []
    unmatched_gt_edge_labels = []
    matched_pred_edges = set() # (u, v, k)
    
    gt_edges_all = list(Ggt.edges(keys=True, data=True))
    pred_edges_all = list(Gpred.edges(keys=True, data=True))
    
    # Build index for faster lookup
    # key: (u_pred, v_pred) -> list of (u, v, k, data)
    pred_edges_by_nodes = defaultdict(list)
    for u, v, k, d in pred_edges_all:
        pred_edges_by_nodes[(u, v)].append((u, v, k, d))
    
    # Iterate over GT edges
    for u_gt, v_gt, k_gt, d_gt in gt_edges_all:
        # Check if endpoints are matched
        if u_gt in node_mapping and v_gt in node_mapping:
            u_pred = node_mapping[u_gt]
            v_pred = node_mapping[v_gt]
            
            # Potential candidates in prediction
            candidates = pred_edges_by_nodes.get((u_pred, v_pred), [])
            
            # Simple greedy match for edges between matched nodes
            # (Could be upgraded to Hungarian if multiple edges exist between same node pair)
            best_sim = -1.0
            best_cand_idx = -1
            
            if candidates:
                for i, (u_p, v_p, k_p, d_p) in enumerate(candidates):
                    # Ideally we don't reuse edges, but simple MultiDiGraph logic:
                    if (u_p, v_p, k_p) in matched_pred_edges:
                        continue
                        
                    # Compare relation semantics
                    emb1 = d_gt.get('embedding')
                    emb2 = d_p.get('embedding')
                    
                    if emb1 is not None and emb2 is not None:
                        sim = _cosine_similarity(emb1, emb2)
                    else:
                        l1 = d_gt.get('label', '')
                        l2 = d_p.get('label', '')
                        sim = 1.0 if l1 == l2 else 0.0
                        
                    if sim > best_sim:
                        best_sim = sim
                        best_cand_idx = i
                
                if best_cand_idx != -1:
                    cand = candidates[best_cand_idx]
                    matched_edge_label_pairs.append((cand[3].get('label', ''), d_gt.get('label', '')))
                    matched_pred_edges.add((cand[0], cand[1], cand[2]))
                else:
                    unmatched_gt_edge_labels.append(((u_gt, v_gt, k_gt), d_gt.get('label', '')))
            else:
                 unmatched_gt_edge_labels.append(((u_gt, v_gt, k_gt), d_gt.get('label', '')))
        else:
             # Endpoints not matched -> Edge cannot be matched
             unmatched_gt_edge_labels.append(((u_gt, v_gt, k_gt), d_gt.get('label', '')))
    
    # Identify unmatched pred edges
    unmatched_pred_edge_labels = []
    for u, v, k, d in pred_edges_all:
        if (u, v, k) not in matched_pred_edges:
            unmatched_pred_edge_labels.append(((u, v, k), d.get('label', '')))
    
    tp = len(matched_edge_label_pairs)
    fp = len(unmatched_pred_edge_labels)
    fn = len(unmatched_gt_edge_labels)
    
    # Calculate Precision, Recall, and F1 (PSG-Score)
    if (tp + fp) == 0 and (tp + fn) == 0:
        precision = 1.0
        recall = 1.0
        f1_score = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (precision + recall) == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
    
    if verbose:
        print("\n--- Evaluation Metrics ---")
        print(f"True Positives (Matched): {tp}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"PSG-Score (F1): {f1_score:.4f}")
    
    return (
        precision,
        recall,
        f1_score,
        matched_edge_label_pairs,
        unmatched_gt_edge_labels,
        unmatched_pred_edge_labels,
        matched_node_pairs,
        unmatched_gt_nodes_list,
        unmatched_pred_nodes_list,
    )



# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Command-line interface for running graph edit distance computation."""
    parser = argparse.ArgumentParser(
        description="Graph Edit Distance using Hungarian algorithm with label and structural similarity."
    )
    parser.add_argument("--g1", type=str, default="graphs/graph_a.json", 
                       help="Path to first graph JSON")
    parser.add_argument("--g2", type=str, default="graphs/graph_b.json", 
                       help="Path to second graph JSON")
    parser.add_argument("--alpha", type=float, default=0.6, 
                       help="Weight for label similarity (semantic: node + edge labels)")
    parser.add_argument("--beta", type=float, default=0.4, 
                       help="Weight for structural similarity (topology)")
    parser.add_argument("--node-sub-cost", type=float, default=1.0, 
                       help="Base cost multiplier for node label substitution")
    parser.add_argument("--edge-sub-cost", type=float, default=0.5, 
                       help="Base cost multiplier for edge label substitution")
    parser.add_argument("--node-del-cost", type=float, default=1.0, 
                       help="Cost to delete a node from G1")
    parser.add_argument("--node-ins-cost", type=float, default=1.0, 
                       help="Cost to insert a node (from G2)")
    parser.add_argument("--edge-del-cost", type=float, default=0.5, 
                       help="Cost to delete an edge")
    parser.add_argument("--edge-ins-cost", type=float, default=0.5, 
                       help="Cost to insert an edge")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", 
                       help="SentenceTransformer model name for computing text embeddings")
    parser.add_argument("--match-threshold", type=float, default=1.5, 
                       help="Max substitution cost to accept a match (higher = more permissive)")
    args = parser.parse_args()

    if args.alpha < 0 or args.beta < 0 or (args.alpha + args.beta) <= 0:
        raise ValueError("alpha and beta must be non-negative and not both zero")

    G1 = load_graph(args.g1)
    G2 = load_graph(args.g2)

    # Initialize sentence-transformer model and compute text embeddings
    model = SentenceTransformer(args.model_name)
    compute_text_embeddings_for_graph(G1, model)
    compute_text_embeddings_for_graph(G2, model)

    C, ids1, ids2 = build_cost_matrix(
        G1, G2, 
        alpha=args.alpha, 
        beta=args.beta,
    )
    
    matched, unmatched_1, unmatched_2 = run_hungarian(
        C, ids1, ids2, 
        node_del_cost=args.node_del_cost,
        node_ins_cost=args.node_ins_cost,
        match_threshold=args.match_threshold
    )

    # Compute full graph edit distance
    ged_result = compute_graph_edit_distance(
        G1, G2, matched, unmatched_1, unmatched_2,
        node_del_cost=args.node_del_cost,
        node_ins_cost=args.node_ins_cost,
        edge_del_cost=args.edge_del_cost,
        edge_ins_cost=args.edge_ins_cost
    )

    # Pretty print results
    print("=" * 70)
    print("GRAPH EDIT DISTANCE RESULTS")
    print("=" * 70)
    print(f"\nTotal Graph Edit Distance: {ged_result['total_cost']:.3f}")
    print(f"\nNode Operations:")
    print(f"  Substitutions: {ged_result['breakdown']['nodes_matched']} "
          f"(cost: {ged_result['node_costs']['substitution']:.3f})")
    print(f"  Deletions:     {ged_result['breakdown']['nodes_deleted']} "
          f"(cost: {ged_result['node_costs']['deletion']:.3f})")
    print(f"  Insertions:    {ged_result['breakdown']['nodes_inserted']} "
          f"(cost: {ged_result['node_costs']['insertion']:.3f})")
    print(f"\nEdge Operations:")
    print(f"  Deletions:     {ged_result['breakdown']['edges_deleted']} "
          f"(cost: {ged_result['edge_costs']['deletion']:.3f})")
    print(f"  Insertions:    {ged_result['breakdown']['edges_inserted']} "
          f"(cost: {ged_result['edge_costs']['insertion']:.3f})")
    
    print(f"\n{'─' * 70}")
    print("Matched Node Pairs (substitutions):")
    print(f"{'─' * 70}")
    for id1, id2, cost in sorted(matched, key=lambda x: x[2]):
        n1 = G1.nodes[id1]
        n2 = G2.nodes[id2]
        print(f"  {id1:10s} ({n1.label:20s}) -> {id2:10s} ({n2.label:20s})  cost={cost:.3f}")
    
    if unmatched_1:
        print(f"\n{'─' * 70}")
        print("Deleted Nodes (from G1):")
        print(f"{'─' * 70}")
        for nid in unmatched_1:
            print(f"  {nid} ({G1.nodes[nid].label})")
    
    if unmatched_2:
        print(f"\n{'─' * 70}")
        print("Inserted Nodes (from G2):")
        print(f"{'─' * 70}")
        for nid in unmatched_2:
            print(f"  {nid} ({G2.nodes[nid].label})")
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
