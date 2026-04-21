
import sys
import os
from pathlib import Path

# Add univlm to path
univlm_path = Path("/sc/home/constantin.auga/MP/univlm")
sys.path.append(str(univlm_path))
sys.path.append(str(univlm_path / "evaluation"))

# Import the module under test
sys.path.append("/sc/home/constantin.auga/MP/vlm-benchmark/univlm_eval")
from generate_images_all_models import generate_images_for_model
from answer_image_questions_all_models import answer_questions_for_model

def test_bagel_integration():
    print("Testing Bagel integration...")
    
    output_dir = Path("test_results_bagel")
    output_dir.mkdir(exist_ok=True)
    
    # Text-to-Image
    print("\n--- Testing Text-to-Image ---")
    prompt = "A red square on a blue background"
    
    try:
        # Use default path (HF repo ID)
        model_type = "bagel"
        # We need to get the default path from config since we don't have a local override
        from generate_images_all_models import MODEL_CONFIGS
        model_path = MODEL_CONFIGS[model_type]['default_path']
        print(f"Using model path: {model_path}")

        result_gen = generate_images_for_model(
            model_type=model_type,
            model_path=model_path,
            config_path=None,
            prompt=prompt,
            output_dir=output_dir,
            device=0,
            seed=42,
            verbose=True
        )
        
        if result_gen["success"]:
            print(f"SUCCESS: Image generated at {result_gen['image_path']}")
            image_path = result_gen['image_path']
        else:
            print(f"FAILURE: {result_gen['error']}")
            # Create dummy image for QA test if generation fails
            image_path = str(output_dir / "dummy.jpg")
            from PIL import Image
            Image.new('RGB', (100, 100), color='blue').save(image_path)
            
    except Exception as e:
        print(f"Error during T2I test: {e}")
        import traceback
        traceback.print_exc()
        # Create dummy image for QA test if generation crashes
        image_path = str(output_dir / "dummy.jpg")
        from PIL import Image
        Image.new('RGB', (100, 100), color='blue').save(image_path)

    # Image-to-Text (QA)
    print("\n--- Testing Image-to-Text (QA) ---")
    image_files = [Path(image_path)]
    question = "Describe the image."
    
    try:
        result_qa = answer_questions_for_model(
            model_type="bagel",
            model_path=model_path,
            config_path=None,
            image_files=image_files,
            question=question,
            output_dir=output_dir,
            device=0,
            seed=42,
            verbose=True
        )
        
        if result_qa["success"]:
            print(f"SUCCESS: Answered {result_qa['num_answered']} questions.")
            print(f"Answers saved to: {result_qa['answers_file']}")
            with open(result_qa['answers_file'], 'r') as f:
                print(f"Content:\n{f.read()}")
        else:
            print(f"FAILURE: {result_qa['error']}")
            
    except Exception as e:
        print(f"Error during QA test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bagel_integration()
