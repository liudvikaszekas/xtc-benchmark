
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

def test_tar_qa():
    print("Testing Tar QA integration...")
    
    output_dir = Path("test_results_tar_qa")
    output_dir.mkdir(exist_ok=True)
    
    # Create a dummy image
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    dummy_image_path = image_dir / "test_image.jpg"
    if not dummy_image_path.exists():
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(dummy_image_path)
    
    image_files = [dummy_image_path]
    question = "What color is the image?"
    
    # Use the local model path since we know it exists and works, 
    # mirroring how we tested generation.
    # The default path in the config is "csuhan/Tar-Lumina2", 
    # which 'tar_roundtrip.py' should handle if configured correctly,
    # but for this test we explicitly pass the existing local path to be safe/fast.
    model_path = "/sc/home/constantin.auga/models/Tar-Lumina2"
    
    try:
        print(f"Using model path: {model_path}")
        
        result = answer_questions_for_model(
            model_type="tar",
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
    test_tar_qa()
