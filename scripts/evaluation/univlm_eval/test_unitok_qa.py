import sys
import os
import json
from pathlib import Path
from PIL import Image

# Add univlm to path
univlm_path = Path("/sc/home/constantin.auga/MP/univlm")
sys.path.append(str(univlm_path))
sys.path.append(str(univlm_path / "evaluation"))

# Import the module under test
sys.path.append("/sc/home/constantin.auga/MP/vlm-benchmark/univlm_eval")
from answer_image_questions_all_models import answer_questions_for_model

def test_unitok_qa():
    print("Testing UniTok QA integration...")
    
    output_dir = Path("test_results_unitok_qa")
    output_dir.mkdir(exist_ok=True)
    
    # Create a dummy image
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    dummy_image_path = image_dir / "test_image.jpg"
    if not dummy_image_path.exists():
        # Create a solid red square
        img = Image.new('RGB', (100, 100), color='red')
        img.save(dummy_image_path)
    
    image_files = [dummy_image_path]
    question = "What color is the image?"
    
    try:
        from answer_image_questions_all_models import MODEL_CONFIGS
        
        # Try to pull the path from config, otherwise fallback to the HF default
        if 'unitok' in MODEL_CONFIGS and 'default_path' in MODEL_CONFIGS['unitok']:
            model_path = MODEL_CONFIGS['unitok']['default_path']
        else:
            model_path = "Alpha-VLLM/Lumina-mGPT-7B-512"
            print("UniTok config not found, falling back to standard path.")
            
        print(f"Using model path: {model_path}")
        
        result = answer_questions_for_model(
            model_type="unitok",
            model_path=model_path,
            config_path=None,
            image_files=image_files,
            question=question,
            output_dir=output_dir,
            device=0,
            seed=42,
            verbose=True
        )
        
        if result["success"]:
            print(f"\nSUCCESS: Answered {result['num_answered']} questions.")
            print(f"Answers saved to: {result['answers_file']}")
            with open(result['answers_file'], 'r') as f:
                print(f"Content:\n{f.read()}")
        else:
            print(f"\nFAILURE: {result['error']}")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unitok_qa()