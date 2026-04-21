#!/usr/bin/env python3
"""
Enhanced question generation utilities for VQA with member_attributes support.

Features:
- Natural text generation for groups and individuals
- Proper handling of member_attributes
- Smart pluralization and agreement
- Answer obfuscation
"""

import re
from typing import Dict, List, Any, Tuple, Optional


class RelationFormatter:
    """Format relation predicates naturally."""
    
    RELATION_MAPPINGS = {
        "on back of": "behind",
        "attached to": "attached to",
        "hanging from": "hanging from",
        "lying on": "on",
        "covering": "covering",
        "painted on": "painted on",
        "leaning on": "leaning on",
        "parked on": "parked on",
        "standing on": "standing on",
        "walking on": "walking on",
        "sitting on": "sitting on",
        "covered in": "covered in",
        "lying in": "lying in",
        "sitting in": "sitting in",
        "standing in": "standing in",
        "walking in": "walking in",
        "flying in": "flying in",
        "growing on": "growing on",
        "walking next to": "walking next to",
        "standing next to": "standing next to",
        "sitting next to": "sitting next to",
        "parked next to": "parked next to",
        "beneath": "beneath",
        "beside": "beside",
        "in front of": "in front of", 
        "inside": "inside",
        "on": "on",
        "over": "over",
        "under": "under",
        "above": "above",
        "below": "below",
        # Common actions
        "eating": "eating",
        "holding": "holding",
        "playing": "playing",
        "carrying": "carrying",
        "looking at": "looking at",
        "riding": "riding",
        "using": "using",
        "wearing": "wearing",
    }

    @staticmethod
    def format_relation(rel: str) -> str:
        """Map technical relation to natural language."""
        return RelationFormatter.RELATION_MAPPINGS.get(rel, rel)


class LabelFormatter:
    """Format object labels naturally."""
    
    @staticmethod
    def format_label(label: str) -> str:
        """Clean and format label string."""
        # Remove common technical suffixes
        clean = label.replace("-merged", "").replace("-other", "")
        
        # Handle specific cases
        if clean.endswith("-wood"):
            clean = "wood " + clean[:-5]
        
        # Replace remaining hyphens with spaces
        clean = clean.replace("-", " ")
        
        return clean.strip()


class AttributeFormatter:
    """Format attribute keys and values naturally."""
    
    # Map attribute keys to readable formats
    ATTR_KEY_NAMES = {
        # Person attributes
        "upper_clothing_type": "upper clothing",
        "upper_clothing_color": "upper clothing color",
        "lower_clothing_type": "lower clothing",
        "lower_clothing_color": "lower clothing color",
        "headwear_eyewear": "headwear or eyewear",
        "hair_color": "hair color",
        "hair_style": "hair style",
        "held_object_type": "held object",
        "held_object_color": "held object color",
        "clothing_type": "clothing",
        "clothing_color": "clothing color",
        "body_position": "position",
        
        # General attributes
        "brand": "brand",
        "material": "material",
        "material_type": "material",
        "color": "color",
        "primary_color": "color",
        "size": "size",
        "pattern": "pattern",
        "pattern_type": "pattern",
        "texture": "texture",
        "state": "state",
        "action": "action",
        
        # Vehicle attributes
        "viewpoint_angle": "viewpoint",
        "text_number_visible": "visible text",
        
        # Traffic / Street / Infrastructure
        "light_color_state": "color",
        "text_on_sign": "text",
        "mounted_position": "position",
        
        # Surface / Object properties
        "surface_material": "surface material",
        "marking_type": "markings",
        "wet_dry_state": "condition",
        
        # Openable / Appliances
        "open_closed_state": "state",
        "door_open_closed": "state",
        "screen_on_off": "state",
        "content_visible": "content",
        
        # Food
        "topping_type": "topping",
        
        # Plants
        "container_type": "container",
        
        # Environment
        "weather_type": "weather",
        "wave_cloud_visible": "visibility of waves or clouds",
        
        # Counts / Quantities
        "window_count": "number of windows",
        "drawer_count": "number of drawers",
        
        # Text / Brand
        "brand_text_visible": "brand",
        "text_visible": "text",
    }
    
    @staticmethod
    def format_key(key: str) -> str:
        """Convert attribute key to readable form."""
        return AttributeFormatter.ATTR_KEY_NAMES.get(key, key.replace("_", " "))
    
    @staticmethod
    def format_value(value: Any) -> str:
        """Format attribute value naturally."""
        if value is None or value == "" or value == [] or value == {}:
            return ""
        
        if isinstance(value, list):
            if not value:
                return ""
            # For lists, join with commas/and
            if len(value) == 1:
                return str(value[0])
            elif len(value) == 2:
                return f"{value[0]} and {value[1]}"
            else:
                return ", ".join(str(v) for v in value[:-1]) + f", and {value[-1]}"
        
        return str(value)
    
    @staticmethod
    def format_attr_phrase(key: str, value: Any) -> str:
        """Format 'key value' as natural phrase.
        
        Examples:
          - color red → red
          - clothing_type shirt → wearing a shirt
          - held_object_type racquet → holding a racquet
          - material_type wood → made of wood
        """
        formatted_key = AttributeFormatter.format_key(key)
        formatted_value = AttributeFormatter.format_value(value)
        
        if not formatted_value:
            return ""
        
        # Action - simple value (usually participle like "walking")
        if key == "action":
             return formatted_value

        # --- Specific Part Handlers (must come before generic) ---
        
        # Clothing / Person parts
        if "clothing" in key and "color" not in key:
            return f"wearing {formatted_value}"
        elif "held_object" in key and "color" not in key:
            return f"holding {formatted_value}"
        elif "headwear" in key or "eyewear" in key:
            return f"wearing {formatted_value}"
        elif "clothing_color" in key:
             type_name = key.replace("_color", "").replace("_", " ")
             return f"wearing {formatted_value} {type_name}"
             
        # Body parts (hair, eyes, skin, etc.)
        elif "hair" in key and "color" in key:
             return f"with {formatted_value} hair"
        elif "eye" in key and "color" in key:
             return f"with {formatted_value} eyes"
        elif "skin" in key:
             return f"with {formatted_value} skin"
        elif "hair" in key and "style" in key:
             return f"with {formatted_value} hair"
        # Catch-all for other hair attributes
        elif "hair" in key:
            return f"with {formatted_value} hair"

        # General Part Colors (e.g., roof_color -> with red roof)
        # Exclude generic object colors
        elif "color" in key and key not in ["color", "primary_color", "light_color_state"]:
             part = key.replace("_color", "").replace("_", " ")
             return f"with {formatted_value} {part}"

        # General Part Materials (e.g., roof_material -> with roof made of wood)
        elif "material" in key and key not in ["material", "surface_material"]:
             part = key.replace("_material", "").replace("_", " ")
             return f"with {part} made of {formatted_value}"

        # --- Generic Attributes (Adjectives or Phrases) ---

        # Physical properties
        elif "color" in key or "light_color_state" in key:
             # Generic color - straight adjective
             return formatted_value 
             
        elif "material" in key or "surface_material" in key:
            return f"made of {formatted_value}"
            
        elif "pattern" in key:
            return f"with a {formatted_value} pattern"
            
        elif "texture" in key:
            return f"with {formatted_value} texture"
            
        # Shape / Size / Age / Gender / Emotion / Brand / Sport
        # These are usually adjectives. Return value directly.
        # ObjectDescriber must handle placing them before noun.
        elif key in ["shape", "size", "age", "sex", "gender", "emotion", "brand", "sport", "condition", "state"]:
             return formatted_value

        # Position / Viewpoint
        elif "position" in key or "viewpoint" in key:
            return f"in {formatted_value} position" if "position" in key else f"seen from the {formatted_value}"
            
        # States (boolean-ish)
        elif "open_closed" in key or "on_off" in key or "wet_dry" in key:
             return formatted_value
            
        # Content / Text
        elif "text" in key:
             return f"that says '{formatted_value}'"
        elif "content" in key:
             return f"containing {formatted_value}"
             
        # Food/Plants
        elif "topping" in key:
             return f"topped with {formatted_value}"
        elif "container" in key:
             return f"in a {formatted_value}"
             
        # Counts
        elif "count" in key:
             thing = key.split('_')[0] if "_" in key else "item"
             suffix = "s" if str(formatted_value) != "1" else ""
             return f"with {formatted_value} {thing}{suffix}"
             
        # Default fallback
        # If we haven't matched anything, returning "key: value" is safest to avoid data loss,
        # but for description quality, maybe we just return value if it looks like an adjective?
        # No, "key: value" is safer for debugging/weird keys.
        return f"{formatted_key}: {formatted_value}"


class ObjectDescriber:
    """Generate natural descriptions of objects/groups."""
    
    @staticmethod
    def describe_single_object(obj: Dict[str, Any], include_attrs: bool = True) -> str:
        """Describe a single object naturally.
        
        Args:
            obj: Object dict with 'label' and 'attrs'
            include_attrs: Whether to include attribute details
        """
        raw_label = obj.get("label", "object")
        label = LabelFormatter.format_label(raw_label)
        
        if not include_attrs:
            return label
        
        attrs = obj.get("attrs", {})
        if not attrs:
            return label
        
        # Categorize attributes into pre-nominal (adjectives) and post-nominal (phrases)
        pre_nominal = []
        post_nominal = []
        
        # Priority/Ordering for processing
        # We can just iterate all keys, but specific order helps naturalness
        
        # Keys that usually give adjectives (colors, states, weather)
        # REMOVED clothing colors from adjectives list to force them into post-nominal phrases
        adj_keys = [
            "primary_color", "color", "light_color_state", 
            "wet_dry_state", "open_closed_state", "screen_on_off", "door_open_closed",
            "weather_type",
            # Common adjectives
            "shape", "size", "age", "sex", "gender", "emotion", "brand", "sport", "condition", "state",
            # Looser state matches (if key is just these words)
            "open_closed", "on_off", "wet_dry", "visibility"
        ]
        
        # Sort keys to ensure deterministic order if not in priority list
        all_keys = sorted(attrs.keys())
        
        for key in all_keys:
            val = attrs[key]
            phrase = AttributeFormatter.format_attr_phrase(key, val)
            if not phrase:
                continue
                
            if key in adj_keys:
                # Adjective - goes before
                pre_nominal.append(phrase)
            else:
                # Phrase - goes after
                post_nominal.append(phrase)
        
        # Construct: [adjectives] [label] [phrases]
        parts = []
        if pre_nominal:
            parts.append(" ".join(pre_nominal))
        
        parts.append(label)
        
        if post_nominal:
            parts.append(" ".join(post_nominal))
            
        return " ".join(parts)
    
    @staticmethod
    def describe_group(members: List[Dict[str, Any]], label: str, 
                       num_to_show: int = 3) -> Tuple[str, List[str]]:
        """Describe a group of objects.
        
        Returns:
            (group_description, member_descriptions)
        """
        count = len(members)
        
        # Group intro
        group_desc = f"a group of {count} {label}s" if count > 1 else f"a {label}"
        
        # Individual member descriptions
        member_descs = []
        for i, member in enumerate(members[:num_to_show]):
            # Use describe_single_object logic for members too
            # Create a localized object dict
            mem_obj = {"label": label, "attrs": member.get("attributes", {})}
            desc = ObjectDescriber.describe_single_object(mem_obj)
            member_descs.append(desc)
        
        return group_desc, member_descs


class QuestionNaturalizer:
    """Transform template questions into natural language."""
    
    @staticmethod
    def instantiate_question(
        template: str,
        replacements: Dict[str, str],
        avoid_answer: Optional[str] = None
    ) -> str:
        """Replace placeholders in template with actual values.
        
        Args:
            template: Question template with <key> placeholders
            replacements: Dict of key -> value replacements
            avoid_answer: Don't include this string in the question
        
        Returns:
            Instantiated question without answer hints
        """
        question = template
        
        # Replace all placeholders
        for key, value in replacements.items():
            if value == avoid_answer:
                # Skip including the answer
                value = "[object]" if "label" in key else "[attribute]"
            
            # Clean labels if this is a label placeholder
            if "label" in key.lower() or key in ["<SUBJ_LABEL>", "<OBJ_LABEL>"]:
                value = LabelFormatter.format_label(str(value))
            
            # Clean relations
            if "rel" in key.lower():
                value = RelationFormatter.format_relation(str(value))
                
            # Clean attribute keys
            if "attr_key" in key.lower() or "query_key" in key.lower() or "filter_key" in key.lower() or "_key" in key.lower():
                value = AttributeFormatter.format_key(str(value))
                
            # Clean attribute values
            if "attr_value" in key.lower() or "filter_value" in key.lower() or "_value" in key.lower():
                # Avoid formatting if it's already a clean string or if we want exact value? 
                # Usually values are "red", "small". format_value handles lists.
                value = AttributeFormatter.format_value(value)

            placeholder = f"<{key}>"
            question = question.replace(placeholder, str(value))
        
        # Clean up any remaining placeholders
        question = re.sub(r"<\w+>", "", question)
        
        # Fix spacing and punctuation
        question = re.sub(r"\s+", " ", question).strip()
        
        # Add question mark if missing and doesn't end in period
        if question and not question.endswith("?") and not question.endswith("."):
            question += "?"
        
        return question
    
    @staticmethod
    def fix_agreement(question: str, subject_count: int) -> str:
        """Fix subject-verb agreement based on count.
        
        Args:
            question: Question text
            subject_count: Number of objects (1 = singular, >1 = plural)
        
        Returns:
            Question with corrected agreement
        """
        if subject_count == 1:
            # Singular
            question = re.sub(r"\b(are|have)\b", lambda m: "is" if m.group(1) == "are" else "has", question)
            question = re.sub(r"\b<label>s\b", "<label>", question)
        else:
            # Plural
            question = re.sub(r"\b(is|has)\b", lambda m: "are" if m.group(1) == "is" else "have", question)
        
        return question
    
    @staticmethod
    def remove_answer_hints(question: str, answer: str) -> str:
        """Remove potential answer hints from question.
        
        Args:
            question: Question text
            answer: The answer we want to hide
        
        Returns:
            Question without answer hints
        """
        if not answer or answer in ("yes", "no", "", None):
            return question
        
        # Case-insensitive answer removal
        answer_escaped = re.escape(str(answer))
        question = re.sub(
            rf"\b{answer_escaped}\b",
            "[hidden]",
            question,
            flags=re.IGNORECASE
        )
        
        return question


class GroupMemberHandler:
    """Handle special logic for group members vs single objects."""
    
    @staticmethod
    def is_group(obj: Dict[str, Any]) -> bool:
        """Check if object is a group member."""
        return obj.get("is_member", False) or "seg_id" in obj
    
    @staticmethod
    def get_member_context(obj: Dict[str, Any]) -> Dict[str, Any]:
        """Get context about which group this member belongs to."""
        return {
            "original_index": obj.get("original_index"),
            "seg_id": obj.get("seg_id"),
            "is_member": obj.get("is_member", False),
        }
    
    @staticmethod
    def build_member_specific_question(
        template: str,
        label: str,
        attrs: Dict[str, Any]
    ) -> str:
        """Build a question specific to a group member."""
        # For group members, we might want simpler, more direct questions
        # since we're asking about specific individuals in a group
        
        simple_templates = {
            "attribute": "What is the <attr_key> of this {label}?",
            "comparison": "How does this {label}'s <attr_key> compare?",
            "existence": "Does this {label} have <attr_key> <value>?",
        }
        
        # Use appropriate template
        if len(attrs) == 0:
            template = simple_templates["existence"]
        else:
            template = simple_templates["attribute"]
        
        return template.format(label=label)


# Convenience functions for common tasks

def format_attr_for_display(key: str, value: Any) -> str:
    """Format attribute for display in questions."""
    return AttributeFormatter.format_attr_phrase(key, value)


def describe_object_for_vqa(obj: Dict[str, Any]) -> str:
    """Get natural description of object for VQA."""
    return ObjectDescriber.describe_single_object(obj, include_attrs=True)


def instantiate_and_clean_question(
    template: str,
    replacements: Dict[str, str],
    answer: Optional[str] = None
) -> str:
    """Full pipeline: instantiate template and clean answer hints."""
    question = QuestionNaturalizer.instantiate_question(
        template, 
        replacements,
        avoid_answer=answer
    )
    
    # Ensure all placeholders are removed
    question = re.sub(r"<\w+>", "", question)
    question = " ".join(question.split()).strip()
    
    if answer:
        question = QuestionNaturalizer.remove_answer_hints(question, answer)
    
    # Ensure ends with punctuation
    if question and not question.endswith(("?", ".")):
        question += "?"
    
    return question


if __name__ == "__main__":
    # Test examples
    
    # Test single object description
    obj = {
        "label": "person",
        "attrs": {
            "upper_clothing_type": "shirt",
            "upper_clothing_color": "blue",
            "hair_color": "brown",
            "held_object_type": "racquet"
        }
    }
    
    print("Single object:")
    print(ObjectDescriber.describe_single_object(obj))
    print()
    
    # Test group description
    members = [
        {
            "attributes": {
                "upper_clothing_type": "shirt",
                "upper_clothing_color": "red"
            }
        },
        {
            "attributes": {
                "upper_clothing_type": "dress",
                "upper_clothing_color": "blue"
            }
        },
        {
            "attributes": {
                "upper_clothing_type": "pants",
                "upper_clothing_color": "black"
            }
        }
    ]
    
    print("Group description:")
    desc, member_descs = ObjectDescriber.describe_group(members, "person", num_to_show=2)
    print(f"Group: {desc}")
    for m in member_descs:
        print(f"  - {m}")
    print()
    
    # Test question instantiation
    template = "What is the <attr_key> of the <label>?"
    replacements = {
        "attr_key": "color",
        "label": "person"
    }
    answer = "red"
    
    print("Question instantiation:")
    q = instantiate_and_clean_question(template, replacements, answer)
    print(f"Question: {q}")
    print(f"Answer: {answer}")
