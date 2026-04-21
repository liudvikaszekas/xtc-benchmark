
import sys
from pathlib import Path

# Add univlm to path
univlm_path = Path("/sc/home/constantin.auga/MP/univlm")
sys.path.append(str(univlm_path))
sys.path.append(str(univlm_path / "evaluation"))

from roundtrip_factory import create_roundtrip_generator

def test_tar():
    print("Testing Tar integration...")
    model_path = "/sc/home/constantin.auga/models/Tar-Lumina2"
    
    try:
        # Create generator
        print(f"Initializing Tar generator from {model_path}...")
        generator = create_roundtrip_generator(
            model_type="tar",
            model_path=model_path,
            device=0
        )
        print("Success: Generator initialized.")
        
        # Test Text-to-Image
        print("\nTesting Text-to-Image...")
        prompt = "A blue circle on a white background"
        image = generator.generate_image_from_text(prompt)
        output_image_path = "test_tar_image.jpg"
        image.save(output_image_path)
        print(f"Success: Image generated and saved to {output_image_path}")
        
        # Test Image-to-Text
        print("\nTesting Image-to-Text...")
        caption = generator.generate_caption_from_image(image, "Describe this image.")
        print(f"Success: Caption generated: {caption}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tar()
