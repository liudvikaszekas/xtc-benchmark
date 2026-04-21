# VQA Question Generation

This tool generates natural language VQA questions from Scene Graph JSON data.

## Question Types
The generator procedurally creates 5 types of questions:

1.  **Attribute Query**: Asking for a specific attribute of an object.
    *   *Example*: "What is the color of the red car?"
2.  **Object Identification**: Identifying an object based on its attributes.
    *   *Example*: "What object is red and metallic?"
3.  **Relationship Query**: Asking about the relationship between two specific objects.
    *   *Example*: "What is the relationship between the person and the car?"
4.  **Counting**: Counting objects of a specific category.
    *   *Example*: "How many cars are there?"
5.  **Count Comparison**: Comparing the counts of two object categories.
    *   *Example*: "Are there more cats than dogs?"

## Usage

Run the `generate_questions.py` script pointing to your scene graph directory.

```bash
generate_questions_from_json.py --attributes_json /path/to/xtc-benchmark/pipeline/run_1000_coco_images/5_attributes_gt/attributes.jsonl --metadata_file meta_data/metadata.json --synonyms_json meta_data/synonyms.json --template_dir templates --output_questions_file output/test_generated_questions.json --graph_merge_dir /path/to/xtc-benchmark/pipeline/run_1000_coco_images/4_graph_merge_gt
```
-> template_dir is legacy, we do not use it at the moment anymore.

## Features
- **Natural Language Formatting**: Uses `vqa_utils.py` to ensure descriptions sound natural (e.g., "person wearing shirt" instead of "person shirt").
- **Disambiguation**: Automatically finds unique attributes to distinguish between multiple objects of the same class.
- **Member Attributes**: Handles hierarchical object representations (groups/members).

## Output
The output is a JSON file containing a list of questions, answers, and the associated functional programs (for neuro-symbolic execution).
