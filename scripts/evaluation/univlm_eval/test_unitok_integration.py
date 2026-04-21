import sys
import os
from pathlib import Path

# Add univlm to path
univlm_path = Path("/sc/home/constantin.auga/MP/univlm")
sys.path.append(str(univlm_path))
sys.path.append(str(univlm_path / "evaluation"))

# Import the actual module we want to test
sys.path.append(str(Path(__file__).resolve().parent))
from generate_images_all_models import generate_images_for_model

def test_unitok_integration():
    print("Testing UniTok integration via generate_images_all_models.py...")
    
    output_dir = Path("test_results_unitok")
    output_dir.mkdir(exist_ok=True)
    
    prompt = "A blue circle on a white background"
    
    try:
        from generate_images_all_models import MODEL_CONFIGS
        
        # Try to pull the path from config, otherwise fallback to the HF default
        if 'unitok' in MODEL_CONFIGS and 'default_path' in MODEL_CONFIGS['unitok']:
            default_path = MODEL_CONFIGS['unitok']['default_path']
            print(f"Default path from config: {default_path}")
        else:
            default_path = "Alpha-VLLM/Lumina-mGPT-7B-768-Omni"
            print(f"UniTok config not found, falling back to standard path: {default_path}")
            
        # You can override this with a local path (e.g., "/sc/home/constantin.auga/models/Lumina-mGPT-7B-768-Omni")
        # if you have already downloaded the weights.
        model_path = default_path
        
        print(f"Using model path: {model_path}")

        result = generate_images_for_model(
            model_type="unitok",
            model_path=model_path,
            config_path=None,
            prompt=prompt,
            output_dir=output_dir,
            device=0,
            seed=42,
            verbose=True
        )
        
        if result["success"]:
            print(f"\nSUCCESS: Image generated at {result['image_path']}")
        else:
            print(f"\nFAILURE: {result['error']}")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unitok_integration()
