#!/usr/bin/env python3
"""
Utilities for generating natural language descriptions of objects, including groups.

Supports describing both individual objects and grouped objects with per-member attributes.
"""

from typing import Dict, List, Any
import json


class GroupPromptGenerator:
    """Generate natural language descriptions for both single and group objects."""
    
    @staticmethod
    def describe_object(box: Dict[str, Any], image_id: int = None, identifier: str = None) -> str:
        """Generate a description for a single object (grouped or not).
        
        Args:
            box: Scene graph box object with label and attributes
            image_id: Optional image ID for context
            identifier: Optional unique identifier (e.g., "Person_1")
            
        Returns:
            Natural language description of the object(s)
        """
        label = box.get("label", "object")
        
        # Check if this is a group object with member_attributes
        member_attrs = box.get("member_attributes", [])
        
        if member_attrs and len(member_attrs) > 1:
            # It's a group - generate group description
            return GroupPromptGenerator._describe_group(label, member_attrs, identifier)
        else:
            # Single object
            attrs = box.get("attributes", {})
            return GroupPromptGenerator._describe_single(label, attrs, identifier)
    
    @staticmethod
    def _describe_group(label: str, member_attrs: List[Dict[str, Any]], identifier: str = None) -> str:
        """Generate description for a group of objects.
        
        Args:
            label: Object label/category (e.g., "person", "car")
            member_attrs: List of {"seg_id": ..., "attributes": {...}} dicts
            identifier: Optional group identifier
            
        Returns:
            Group description with individual member descriptions
        """
        count = len(member_attrs)
        
        # Pluralize label
        plural_label = f"{label}s" if not label.endswith('s') else label
        
        # Start with group intro
        description = f"A group of {count} {plural_label}"
        if identifier:
             description += f" ({identifier})"
        
        # Generate member descriptions if we have detailed info
        member_descriptions = []
        for i, member in enumerate(member_attrs, 1):
            member_attrs_dict = member.get("attributes", {})
            
            # Generate sub-identifier for member (e.g. PersonGroup_1_1)
            sub_identifier = f"{identifier}_{i}" if identifier else None
            
            # Pass sub-identifier to member description
            member_desc = GroupPromptGenerator._describe_single(label, member_attrs_dict, identifier=sub_identifier)
            
            # Only add individual descriptions if they have attributes
            if member_attrs_dict:
                member_descriptions.append(f"{i}) {member_desc}")
        
        if member_descriptions:
            description += ":\n  " + "\n  ".join(member_descriptions)
        else:
            description += " standing together."
        
        return description
    
    @staticmethod
    def _describe_single(label: str, attributes: Dict[str, Any], identifier: str = None) -> str:
        """Generate description for a single object.
        
        Args:
            label: Object label/category
            attributes: Attribute dict mapping attribute names to values
            identifier: Optional unique identifier string
            
        Returns:
            Natural language description
        """
        desc = label
        
        # Append ID information if available
        if identifier:
             desc += f" ({identifier})"
        
        if not attributes:
            return desc
        
        # Create attribute phrases
        attr_phrases = []
        
        # Process attributes in preferred order
        attr_order = [
            "color", "size", "material", "state", "pattern", "texture",
            "clothing_type", "clothing_color", "upper_clothing_type", "upper_clothing_color",
            "lower_clothing_type", "lower_clothing_color", "hair_color", "action"
        ]
        
        for attr_type in attr_order:
            if attr_type in attributes:
                val = attributes[attr_type]
                phrase = GroupPromptGenerator._format_attribute(attr_type, val)
                if phrase:
                    attr_phrases.append(phrase)
        
        # Add any remaining attributes not in standard order
        used_keys = set(attr_order)
        # Explicitly exclude attributes that shouldn't be in the description
        excluded_keys = {"visual_reasoning", "confidence", "score"}
        
        for key in sorted(attributes.keys()):
            if key not in used_keys and key not in excluded_keys:
                val = attributes[key]
                phrase = GroupPromptGenerator._format_attribute(key, val)
                if phrase:
                    attr_phrases.append(phrase)
        
        if attr_phrases:
            return desc + " " + " ".join(attr_phrases)
        else:
            return desc
    
    @staticmethod
    def _format_attribute(attr_type: str, value: Any) -> str:
        """Convert attribute to natural language phrase.
        
        Args:
            attr_type: Attribute type name
            value: Attribute value (str, list, or None)
            
        Returns:
            Natural language phrase or empty string if no value
        """
        if value is None or value == "":
            return ""
        
        # Handle list values
        if isinstance(value, list):
            if not value:
                return ""
            # Join list values with "and"
            if len(value) == 1:
                value_str = value[0]
            elif len(value) == 2:
                value_str = f"{value[0]} and {value[1]}"
            else:
                value_str = ", ".join(value[:-1]) + f", and {value[-1]}"
        else:
            value_str = str(value)
        
        # Create natural phrase based on attribute type
        if "color" in attr_type.lower():
            return f"in {value_str}"
        elif "clothing_type" in attr_type.lower() or "clothing" in attr_type.lower():
            return f"wearing {value_str}"
        elif attr_type.lower() in ["size"]:
            return f"that is {value_str}"
        elif attr_type.lower() in ["material"]:
            return f"made of {value_str}"
        elif attr_type.lower() in ["state", "action"]:
            return f"that is {value_str}"
        elif attr_type.lower() in ["pattern"]:
            return f"with {value_str} pattern"
        elif attr_type.lower() in ["texture"]:
            return f"with {value_str} texture"
        else:
            # Generic fallback
            attr_readable = attr_type.replace("_", " ")
            return f"with {attr_readable}: {value_str}"


def process_scene_graph_for_prompts(
    scene_graph: Dict[str, Any],
    include_relationships: bool = False
) -> Dict[str, Any]:
    """Convert scene graph objects to natural language descriptions.
    
    Args:
        scene_graph: Scene graph dict with "boxes" and optional "relationships"
        include_relationships: Whether to include relationship descriptions
        
    Returns:
        Dict with:
        - descriptions: List of object descriptions, one per object
        - prompt: Full scene description as single string
        - object_descriptions: List of (index, description) tuples
    """
    descriptions = []
    object_descriptions = {}  # Changed from list to dict
    image_id = scene_graph.get("image_id")
    
    # Generate unique semantic IDs
    box_semantic_ids = {} # idx -> "Person_1"
    label_counters = {}
    
    for idx, box in enumerate(scene_graph.get("boxes", [])):
        raw_label = box.get("label", "object")
        member_attrs = box.get("member_attributes", [])
        is_group = member_attrs and len(member_attrs) > 1

        # Sanitize label
        base_label_key = raw_label.lower().replace(" ", "_")
        
        # Determine counter key based on group status to maintain separate counts
        if is_group:
            label_key = f"{base_label_key}_group"
        else:
            label_key = base_label_key

        # Increment counter
        label_counters[label_key] = label_counters.get(label_key, 0) + 1
        
        # Create concise semantic ID (e.g., Person_1 or PersonGroup_1)
        base_name = raw_label.replace(' ', '_').capitalize()
        if is_group:
            semantic_id = f"{base_name}Group_{label_counters[label_key]}"
        else:
            semantic_id = f"{base_name}_{label_counters[label_key]}"
            
        box_semantic_ids[idx] = semantic_id

    for idx, box in enumerate(scene_graph.get("boxes", [])):
        identifier = box_semantic_ids[idx]
        desc = GroupPromptGenerator.describe_object(box, image_id, identifier)
        descriptions.append(desc)
        object_descriptions[str(idx)] = desc  # Store as dict with str keys
    
    # Create full prompt
    if descriptions:
        prompt_parts = ["Scene with: " + ", ".join(descriptions)]
    else:
        prompt_parts = ["Empty scene."]
    
    result = {
        "descriptions": descriptions,
        "object_descriptions": object_descriptions,
        "image_id": image_id,
        "file_name": scene_graph.get("file_name", "")
    }
    
    # Optionally add relationships
    relationships = scene_graph.get("relations", scene_graph.get("relationships", []))
    
    if include_relationships and relationships:
        # Build map: seg_id -> specific semantic identifier (e.g. 2396745 -> PersonGroup_1_2)
        # Also map box_idx -> general semantic identifier (e.g. 0 -> PersonGroup_1) as fallback
        
        seg_id_to_sub_id = {}
        
        for idx in box_semantic_ids:
             semantic_id = box_semantic_ids[idx]
             box = scene_graph.get("boxes")[idx]
             
             # Map group members
             member_attrs = box.get("member_attributes", [])
             if member_attrs and len(member_attrs) > 1:
                # Group with individual members
                for i, member in enumerate(member_attrs, 1):
                    mid = member.get("seg_id")
                    if mid is not None:
                        # e.g. PersonGroup_1_2
                        seg_id_to_sub_id[mid] = f"{semantic_id}_{i}"
             else:
                # Flat box: map its seg_ids (list or scalar) to the main ID
                seg_ids = box.get("seg_ids", [])
                if isinstance(seg_ids, list):
                    for sid in seg_ids:
                        seg_id_to_sub_id[sid] = semantic_id
                elif isinstance(seg_ids, (int, str)):
                    seg_id_to_sub_id[seg_ids] = semantic_id
                    
        rel_descriptions = []

        for rel in relationships:
            # Handle various key formats
            subj_idx = rel.get("subject_index", rel.get("subject_id", rel.get("subject")))
            obj_idx = rel.get("object_index", rel.get("object_id", rel.get("object")))
            rel_type = rel.get("predicate", rel.get("relation", "related to"))
            
            # Additional IDs for specific members if available
            subj_seg_id = rel.get("subject_id")  # Specific seg_id output by Step 3
            obj_seg_id = rel.get("object_id")    # Specific seg_id output by Step 3

            if isinstance(subj_idx, int) and isinstance(obj_idx, int):
                # Try specific member first (if Step 3 provided 'subject_id' key with raw seg_id)
                subj_desc = seg_id_to_sub_id.get(subj_seg_id)
                # Fallback to general group ID if specific member not found/not applicable
                if not subj_desc:
                    # Try using the first ID from the list as a fallback if available
                    subj_list = rel.get("subject_seg_ids")
                    if subj_list and len(subj_list) > 0:
                        subj_desc = seg_id_to_sub_id.get(subj_list[0])

                    if not subj_desc and subj_idx in box_semantic_ids:
                        subj_desc = box_semantic_ids[subj_idx]
                
                # Same for object
                obj_desc = seg_id_to_sub_id.get(obj_seg_id)
                if not obj_desc:
                    # Try using the first ID from the list as a fallback
                    obj_list = rel.get("object_seg_ids")
                    if obj_list and len(obj_list) > 0:
                        obj_desc = seg_id_to_sub_id.get(obj_list[0])

                    if obj_idx in box_semantic_ids:
                        obj_desc = box_semantic_ids[obj_idx]
                
                if subj_desc and obj_desc:
                    rel_desc = f"{subj_desc} {rel_type} {obj_desc}"
                    rel_descriptions.append(rel_desc)
        
        if rel_descriptions:
            result["relationships"] = rel_descriptions
            prompt_parts.append("Relationships: " + "; ".join(rel_descriptions))
    
    result["prompt"] = " ".join(prompt_parts)
    
    return result


def validate_member_attributes(scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and analyze member_attributes in a scene graph.
    
    Args:
        scene_graph: Scene graph dict
        
    Returns:
        Dict with validation results
    """
    stats = {
        "total_objects": 0,
        "objects_with_attributes": 0,
        "group_objects": 0,
        "group_objects_with_members": 0,
        "total_group_members": 0,
        "issues": []
    }
    
    for box in scene_graph.get("boxes", []):
        stats["total_objects"] += 1
        
        if box.get("attributes"):
            stats["objects_with_attributes"] += 1
        
        member_attrs = box.get("member_attributes", [])
        if member_attrs and len(member_attrs) > 1:
            stats["group_objects"] += 1
            stats["group_objects_with_members"] += 1
            stats["total_group_members"] += len(member_attrs)
        
        # Check for consistency issues
        seg_ids = box.get("seg_ids", [])
        if member_attrs and len(member_attrs) != len(seg_ids):
            stats["issues"].append(
                f"Box {box.get('index')}: {len(member_attrs)} member_attributes but {len(seg_ids)} seg_ids"
            )
    
    return stats


if __name__ == "__main__":
    # Test example
    test_group = {
        "label": "person",
        "bbox_xyxy": [0, 0, 640, 480],
        "member_attributes": [
            {
                "seg_id": 1,
                "attributes": {
                    "upper_clothing_type": "T-shirt",
                    "upper_clothing_color": "white",
                    "hair_color": "brown"
                }
            },
            {
                "seg_id": 2,
                "attributes": {
                    "upper_clothing_type": "dress",
                    "upper_clothing_color": "red",
                    "hair_color": "blonde"
                }
            },
            {
                "seg_id": 3,
                "attributes": {
                    "upper_clothing_type": "jacket",
                    "upper_clothing_color": "blue",
                }
            }
        ]
    }
    
    desc = GroupPromptGenerator.describe_object(test_group, 1000)
    print("Group Description:")
    print(desc)
    print()
    
    # Test single object
    test_single = {
        "label": "car",
        "attributes": {
            "color": ["red", "metallic"],
            "material": "metal",
            "size": "large"
        }
    }
    
    desc = GroupPromptGenerator.describe_object(test_single)
    print("Single Object Description:")
    print(desc)
    print()
    
    # Test full scene
    test_scene = {
        "image_id": 1000,
        "file_name": "000000001000.jpg",
        "boxes": [test_group, test_single]
    }
    
    result = process_scene_graph_for_prompts(test_scene)
    print("Full Scene Prompt:")
    print(result["prompt"])
