"""
Caption Generator from Existing Scene Graphs
Uses the PSGEval method from "Leveraging Panoptic Scene Graph for Evaluating Fine-Grained Text-to-Image Generation"
"""

import json
import networkx as nx
from gas.captions_generation.scene_graph import get_sg_desc
from gas.captions_generation.utils import convert_sg_to_json


def extract_attributes_from_description(description):
    """
    Extract key attributes from the detailed description.
    This is a simple implementation - you can make it more sophisticated.
    """
    attributes = {}
    
    if description:
        description_lower = description.lower()
        
        colors = ['brown', 'tan', 'cream', 'black', 'white', 'dark', 'light', 'beige', 'gray']
        for color in colors:
            if color in description_lower:
                attributes['color'] = color
                break
        
        if 'large' in description_lower or 'big' in description_lower:
            attributes['size'] = 'large'
        elif 'small' in description_lower:
            attributes['size'] = 'small'
        
        if 'rough' in description_lower:
            attributes['texture'] = 'rough'
        elif 'smooth' in description_lower:
            attributes['texture'] = 'smooth'
    
    return attributes


def convert_scene_graph_to_networkx(scene_graph_data):
    """
    Convert your scene graph format to NetworkX DiGraph format
    required by the PSGEval method.
    
    Args:
        scene_graph_data: Dict with 'boxes' and 'relations' keys
        
    Returns:
        NetworkX DiGraph in the format expected by get_sg_desc()
    """
    G = nx.DiGraph()
    
    for box in scene_graph_data.get("boxes", []):
        obj_id = f"object_{box['index'] + 1}"
        
        G.add_node(obj_id, type="object_node", value=box["label"])
        
        # Handle attributes in two formats:
        # 1. Structured format: {"attributes": {"color": ["red"], "size": ["large"]}}
        # 2. Description format: {"description": "a red large object"}
        attributes = {}
        
        if "attributes" in box and isinstance(box["attributes"], dict):
            # Structured attributes from our workflow
            for attr_type, attr_values in box["attributes"].items():
                if isinstance(attr_values, list) and attr_values:
                    # Take the first value for simplicity, or join them
                    attributes[attr_type] = attr_values[0] if len(attr_values) == 1 else ", ".join(attr_values[:3])
                elif isinstance(attr_values, str):
                    attributes[attr_type] = attr_values
        elif "description" in box and box["description"]:
            # Legacy description format
            attributes = extract_attributes_from_description(box["description"])
        
        for attr_idx, (attr_type, attr_value) in enumerate(attributes.items(), 1):
            attr_node_id = f"attribute|{box['index'] + 1}|{attr_idx}"
            G.add_node(
                attr_node_id,
                type="attribute_node",
                value_type=attr_type,
                value=attr_value
            )
            G.add_edge(obj_id, attr_node_id, type="attribute_edge")
    
    # Strictly use "relations" field - no fallback to "top_relations"
    # This ensures consistency and fails fast if scene graphs have unexpected structure
    if "relations" not in scene_graph_data:
        raise KeyError(
            f"Scene graph missing required 'relations' field. "
            f"Available fields: {list(scene_graph_data.keys())}"
        )
    
    relations = scene_graph_data["relations"]
    
    for rel in relations:
        if rel.get("no_relation_score", 0) > rel.get("predicate_score", 0):
            continue
        
        source_id = f"object_{rel['subject_index'] + 1}"
        target_id = f"object_{rel['object_index'] + 1}"
        
        G.add_edge(
            source_id,
            target_id,
            type="relation_edge",
            value_type="spatial",
            value=rel["predicate"]
        )
    
    return G


def generate_caption_from_scene_graph(scene_graph_data):
    """
    Generate a natural language caption from the scene graph.
    
    Args:
        scene_graph_data: Your scene graph dict
        
    Returns:
        tuple: (caption string, networkx graph in JSON format)
    """
    G = convert_scene_graph_to_networkx(scene_graph_data)
    
    caption = get_sg_desc(G)
    visualize_scene_graph(G)
    
    sg_json = convert_sg_to_json(G)
    
    return caption, sg_json


def load_scene_graph_from_file(filepath):
    """Load scene graph from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_caption_output(caption, sg_json, output_path):
    """Save the generated caption and scene graph to file."""
    output = {
        "caption": caption,
        "scene_graph": sg_json
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved output to {output_path}")

def visualize_scene_graph(G, output_path="scene_graph_visualization.png"):
    """
    Create a visual representation of the scene graph using matplotlib.
    
    Args:
        G: NetworkX DiGraph
        output_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    plt.figure(figsize=(14, 10))
    
    object_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'object_node']
    attribute_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'attribute_node']
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=object_nodes,
                          node_color='lightblue',
                          node_size=3000,
                          node_shape='o')
    
    nx.draw_networkx_nodes(G, pos,
                          nodelist=attribute_nodes,
                          node_color='lightgreen',
                          node_size=2000,
                          node_shape='s')
    
    attribute_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'attribute_edge']
    relation_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'relation_edge']
    
    nx.draw_networkx_edges(G, pos,
                          edgelist=attribute_edges,
                          edge_color='gray',
                          style='dashed',
                          arrows=True,
                          arrowsize=20,
                          width=2)
    
    nx.draw_networkx_edges(G, pos,
                          edgelist=relation_edges,
                          edge_color='red',
                          style='solid',
                          arrows=True,
                          arrowsize=20,
                          width=3)
    
    labels = {}
    for node, data in G.nodes(data=True):
        if data['type'] == 'object_node':
            labels[node] = f"{data['value']}"
        elif data['type'] == 'attribute_node':
            labels[node] = f"{data.get('value_type', '')}:\n{data['value']}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if data['type'] == 'relation_edge':
            edge_labels[(u, v)] = data['value']
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, font_color='red')
    
    object_patch = mpatches.Patch(color='lightblue', label='Objects')
    attribute_patch = mpatches.Patch(color='lightgreen', label='Attributes')
    relation_line = mpatches.Patch(color='red', label='Relations')
    attribute_line = mpatches.Patch(color='gray', label='Has Attribute')
    
    plt.legend(handles=[object_patch, attribute_patch, relation_line, attribute_line],
              loc='upper left', fontsize=12)
    
    plt.title("Scene Graph Visualization", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    # plt.show()
    plt.close()

if __name__ == "__main__":
    scene_graph_data = {
        "image_id": 289586,
        "file_name": "images/000000289586.jpg",
        "boxes": [
            {
                "index": 0,
                "label": "rock-merged",
                "bbox_xyxy": [0.0, 0.0, 425.0, 639.0],
                "description": "The visible object in the second image is a large, rugged rock formation. It has a rough, uneven surface with a predominantly brownish-tan color, featuring variations in shade that suggest natural weathering and texture."
            },
            {
                "index": 1,
                "label": "giraffe",
                "bbox_xyxy": [94.0, 96.0, 425.0, 639.0],
                "description": "A giraffe's head and neck are shown in profile, extending diagonally upward from the bottom right toward the top left. The giraffe's coat features a distinctive pattern of large, irregular, polygonal brown patches separated by thin, cream-colored lines."
            }
        ],
        "relations": [
            {
                "subject_index": 0,
                "subject_label": "rock-merged",
                "object_index": 1,
                "object_label": "giraffe",
                "predicate": "beside",
                "predicate_index": 2,
                "predicate_score": 0.2793293595314026,
                "no_relation_score": 0.5532925128936768
            },
            {
                "subject_index": 1,
                "subject_label": "giraffe",
                "object_index": 0,
                "object_label": "rock-merged",
                "predicate": "in front of",
                "predicate_index": 1,
                "predicate_score": 0.4648173451423645,
                "no_relation_score": 0.3535902798175812
            }
        ],
        "top_relations": [
            {
                "subject_index": 1,
                "subject_label": "giraffe",
                "object_index": 0,
                "object_label": "rock-merged",
                "predicate": "in front of",
                "predicate_index": 1,
                "predicate_score": 0.4648173451423645,
                "no_relation_score": 0.3535902798175812
            }
        ]
    }
    
    caption, sg_json = generate_caption_from_scene_graph(scene_graph_data)
    
    print("="*60)
    print("Generated Caption:")
    print("="*60)
    print(caption)
    print("="*60)
    
    save_caption_output(caption, sg_json, "output_caption.json")