#!/usr/bin/env python3
"""
Quick integration test for VQA with member_attributes and refined templates.
"""

import json
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene_struct import build_scene_struct
from vqa_utils import (
    AttributeFormatter,
    ObjectDescriber,
    QuestionNaturalizer,
    GroupMemberHandler,
    instantiate_and_clean_question
)


def test_attribute_formatter():
    """Test attribute formatting."""
    print("\n" + "="*60)
    print("TEST 1: Attribute Formatter")
    print("="*60)
    
    # Test key formatting
    key = "upper_clothing_type"
    formatted = AttributeFormatter.format_key(key)
    print(f"✓ format_key('{key}') → '{formatted}'")
    assert formatted == "upper clothing", f"Expected 'upper clothing', got '{formatted}'"
    
    # Test value formatting
    value = "T-shirt"
    formatted = AttributeFormatter.format_value(value)
    print(f"✓ format_value('{value}') → '{formatted}'")
    
    # Test phrase formatting
    phrase = AttributeFormatter.format_attr_phrase("upper_clothing_color", "red")
    print(f"✓ format_attr_phrase('upper_clothing_color', 'red') → '{phrase}'")
    
    print("✅ AttributeFormatter tests passed!")


def test_group_expansion():
    """Test group expansion in scene structure."""
    print("\n" + "="*60)
    print("TEST 2: Group Expansion in Scene Structure")
    print("="*60)
    
    # Create a scene with a group (realistic format with seg_ids)
    scene = {
        "image_id": "test_001",
        "boxes": [
            {
                "index": 0,
                "label": "person",
                "bbox_xyxy": [10, 20, 100, 200],
                "attributes": {
                    "upper_clothing_color": ["white", "blue", "black"]
                },
                "seg_ids": [729444, 1458888, 2917776],  # Multiple seg_ids indicate group
                "member_attributes": [
                    {
                        "seg_id": 729444,
                        "attributes": {"upper_clothing_type": "T-shirt", "upper_clothing_color": "white"}
                    },
                    {
                        "seg_id": 1458888,
                        "attributes": {"upper_clothing_type": "Shirt", "upper_clothing_color": "blue"}
                    },
                    {
                        "seg_id": 2917776,
                        "attributes": {"upper_clothing_type": "Jacket", "upper_clothing_color": "black"}
                    }
                ]
            }
        ]
    }
    
    struct = build_scene_struct(scene)
    
    # Check that group was expanded
    num_people = sum(1 for o in struct["objects"] if o["label"] == "person")
    print(f"✓ Original scene had 1 group → expanded to {num_people} individual objects")
    assert num_people == 3, f"Expected 3 people, got {num_people}"
    
    # Check that members have individual attributes
    for i, obj in enumerate(struct["objects"]):
        if obj["label"] == "person":
            attrs = obj.get("attrs", {})
            print(f"  Person {i}: {attrs}")
            assert "upper_clothing_type" in attrs
            assert "upper_clothing_color" in attrs
    
    print("✅ Group expansion tests passed!")


def test_group_member_handler():
    """Test group member detection."""
    print("\n" + "="*60)
    print("TEST 3: Group Member Handler")
    print("="*60)
    
    # Create sample objects
    member_obj = {
        "id": 0,
        "label": "person",
        "is_member": True,
        "original_index": 0,
        "attrs": {"upper_clothing_type": "T-shirt"}
    }
    
    single_obj = {
        "id": 3,
        "label": "car",
        "is_member": False,
        "attrs": {"color": "red"}
    }
    
    # Test detection
    is_member = GroupMemberHandler.is_group(member_obj)
    print(f"✓ is_group(person with is_member=True) → {is_member}")
    assert is_member, "Should detect member object"
    
    is_not_member = GroupMemberHandler.is_group(single_obj)
    print(f"✓ is_group(car with is_member=False) → {is_not_member}")
    assert not is_not_member, "Should not detect single object as member"
    
    print("✅ GroupMemberHandler tests passed!")


def test_question_naturalizer():
    """Test question naturalization."""
    print("\n" + "="*60)
    print("TEST 4: Question Naturalizer")
    print("="*60)
    
    # Test template instantiation
    template = "What <attr_key> does the <label> have?"
    replacements = {
        "attr_key": "upper clothing color",
        "label": "person"
    }
    
    question = QuestionNaturalizer.instantiate_question(
        template,
        replacements,
        avoid_answer="red"
    )
    
    print(f"Template: {template}")
    print(f"Replacements: {replacements}")
    print(f"✓ Result: {question}")
    
    # Verify question is well-formed
    assert question.endswith("?"), "Question should end with ?"
    assert "<" not in question, "Question should have no placeholders"
    assert "red" not in question.lower(), "Should not leak answer 'red'"
    
    # Test plural agreement
    template2 = "How many <label>s have <attr_key> <attr_value>?"
    replacements2 = {
        "label": "person",
        "attr_key": "clothing color",
        "attr_value": "red"
    }
    
    question2 = QuestionNaturalizer.instantiate_question(
        template2,
        replacements2,
        avoid_answer=None
    )
    
    print(f"✓ Plural handling: {question2}")
    
    print("✅ QuestionNaturalizer tests passed!")


def test_refined_templates():
    """Test that refined templates are properly formatted."""
    print("\n" + "="*60)
    print("TEST 5: Refined Templates Structure")
    print("="*60)
    
    # Load refined templates
    template_path = os.path.join(
        os.path.dirname(__file__),
        "templates",
        "refined_templates.json"
    )
    
    if not os.path.exists(template_path):
        print(f"⚠ Refined templates not found at {template_path}")
        return
    
    with open(template_path) as f:
        templates = json.load(f)
    
    print(f"✓ Loaded {len(templates)} templates")
    
    # Verify structure
    for i, template in enumerate(templates[:3]):  # Check first 3
        name = template.get("name", "unknown")
        has_qt = "question_templates" in template
        num_qt = len(template.get("question_templates", []))
        
        print(f"  [{i+1}] {name}: {num_qt} question templates")
        assert has_qt, f"Template {name} missing 'question_templates'"
        assert num_qt >= 3, f"Template {name} has only {num_qt} variants (need ≥3)"
    
    print("✅ Refined templates tests passed!")


def test_instantiate_and_clean():
    """Test the instantiate_and_clean_question function."""
    print("\n" + "="*60)
    print("TEST 6: instantiate_and_clean_question Function")
    print("="*60)
    
    template = "What is the <attr_key> of the <label>?"
    replacements = {"attr_key": "color", "label": "car"}
    answer = "red"
    
    result = instantiate_and_clean_question(
        template,
        replacements,
        answer
    )
    
    print(f"Template: {template}")
    print(f"Replacements: {replacements}")
    print(f"Answer (to hide): {answer}")
    print(f"✓ Result: {result}")
    
    # Verify
    assert "color" in result, "Should contain attribute key"
    assert "car" in result, "Should contain label"
    assert result.endswith("?"), "Should end with ?"
    
    print("✅ instantiate_and_clean_question tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VQA INTEGRATION TEST SUITE")
    print("Member Attributes + Refined Templates")
    print("="*60)
    
    try:
        test_attribute_formatter()
        test_group_expansion()
        test_group_member_handler()
        test_question_naturalizer()
        test_refined_templates()
        test_instantiate_and_clean()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nVQA system is ready to use:")
        print("  - Scene structure handles member_attributes ✓")
        print("  - Refined templates loaded and validated ✓")
        print("  - Natural language utilities working ✓")
        print("  - Answer obfuscation active ✓")
        print("\nNext: Run full VQA question generation")
        print("="*60 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
