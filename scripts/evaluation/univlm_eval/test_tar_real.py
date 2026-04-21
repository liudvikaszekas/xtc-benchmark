
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

def test_tar_integration():
    print("Testing Tar integration via generate_images_all_models.py...")
    
    output_dir = Path("test_results_tar")
    output_dir.mkdir(exist_ok=True)
    
    prompt = "A blue circle on a white background"
    
    # We use the default path defined in the module (which we just set to csuhan/Tar-Lumina2)
    # But since we might want to override it to the local cached path to be safe/fast if the user has it,
    # or just trust the default.
    # The user's command showed: /sc/home/constantin.auga/models/Tar-Lumina2
    # So let's try to run it with the explicit local path that we know works, 
    # OR rely on the HF path handling in tar_roundtrip.py.
    
    # Given the user wants "implicit paths like the others", let's try to use the default configuration
    # but pointing to the local path if the HF download logic isn't perfect, 
    # OR just use the 'tar' key and let it resolve.
    
    # However, to be robust for this test run, I will pass the local path 
    # because I know where it is, while the code defaults to the HF ID.
    # Wait, the user said "tar works with ... /models/Tar-Lumina2".
    
    try:
        # Use the function from generate_images_all_models.py
        # We pass model_path=None to force it to look up the default from MODEL_CONFIGS if possible,
        # but generate_images_for_model requires model_path argument.
        # Let's look at generate_images_all_models signature.
        
        # In generate_images_all_models.py:
        # def generate_images_for_model(model_type: str, model_path: str, ...)
        
        # So we must provide a path.
        # If we want to test the configuration variable, we should access MODEL_CONFIGS['tar']['default_path']
        
        from generate_images_all_models import MODEL_CONFIGS
        default_path = MODEL_CONFIGS['tar']['default_path']
        print(f"Default path from config: {default_path}")
        
        # Check if we should override with local path for this test run to ensure it works
        # The user has /sc/home/constantin.auga/models/Tar-Lumina2
        # If default_path is 'csuhan/Tar-Lumina2', the 'tar_roundtrip.py' needs to handle it.
        # I previously updated 'tar_roundtrip.py' to handle checking existence or downloading.
        # But 'tar_roundtrip.py' logic I added:
        
        # if self.t2i_config.lumina2_path == "csuhan/Tar-Lumina2":
        #      try: snapshot_download...
        
        # Ideally, we want to point to the local model if it exists to avoid redownloading.
        
        # Let's try running with the expected default path 'csuhan/Tar-Lumina2' 
        # which will trigger the logic in tar_roundtrip.py.
        # AND let's also support the local path if the user prefers that. 
        
        # For this test, I will use the path from the user's success command:
        # /sc/home/constantin.auga/models/Tar-Lumina2
        # Just to prove the integration works. 
        # Wait, the user asked for "default_path": "csuhan/..."
        
        # Let's test with the local path first to verify the code structure works.
        
        model_path = "/sc/home/constantin.auga/models/Tar-Lumina2"
        
        print(f"Using model path: {model_path}")

        result = generate_images_for_model(
            model_type="tar",
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
    test_tar_integration()
