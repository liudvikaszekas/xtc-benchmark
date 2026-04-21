#!/usr/bin/env python3
"""
Minimal caller script for image question answering.

Usage:
    python call_answer_questions.py
"""

from answer_image_questions_all_models import answer_questions_for_all_models

# Example: Answer a list of questions about images
questions = [
    ("1", "What is in this image?"),
    ("2", "Describe the main objects"),
    ("3", "What colors are prominent?"),
]

results = answer_questions_for_all_models(
    image_dir="../../test_images",
    questions=questions,
    output_dir="qa_results",
    models=["mmada", "blip3o","showo2","showo","januspro", "omnigen2"],
    batch_size=1,
    seed=42,
    verbose=True
)

# Access results
print(f"\nQuestion answering completed!")
print(f"Total questions: {results['num_questions']}")
print(f"Total images: {results['num_images']}")
print(f"Total models: {results['num_models']}")
print(f"Successful: {results['successful_models']}")

# Print per-model results with JSONL paths
for result in results['results']:
    if result['success']:
        print(f"✓ {result['model_name']}: {result['num_answered']} answers → {result['jsonl_path']}")
    else:
        print(f"✗ {result['model_name']}: {result['error']}")
