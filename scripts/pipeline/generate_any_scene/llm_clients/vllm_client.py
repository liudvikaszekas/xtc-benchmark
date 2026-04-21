"""
VLLM Client for Vision-Language Models
Uses the same VLLM setup as attribute generation (Qwen3-VL-32B)
"""

import io
import base64
from typing import List, Dict, Any
from PIL import Image

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:
    print("WARNING: VLLM library not found. Install with 'pip install vllm'")
    LLM = None
    SamplingParams = None

from .base_client import LLMClient


class VLLMClient(LLMClient):
    """VLLM client for Qwen3-VL-32B or similar vision-language models."""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        num_gpus: int = 4,
        max_tokens: int = 100,
        temperature: float = 0.3
    ):
        """
        Initialize VLLM model.
        
        Args:
            model_name: HuggingFace model name
            num_gpus: Number of GPUs to use
            max_tokens: Max tokens for generation
            temperature: Sampling temperature
        """
        if LLM is None:
            raise RuntimeError("VLLM not installed")
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"Initializing VLLM model: {model_name} on {num_gpus} GPUs...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.95,
            max_model_len=8192,  # Same as attribute generation
            limit_mm_per_prompt={"image": 10},
            max_num_seqs=5,  # SAME AS ATTRIBUTE GENERATION - key for OOM avoidance!
            dtype="bfloat16",  # Same as attribute generation
        )
        print("✓ VLLM model initialized")
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>", "\n\n", ".", "!"]
        )
    
    def _format_attributes(self, attributes: Dict[str, List[str]]) -> str:
        """Format attributes into a readable string."""
        if not attributes:
            return "no specific attributes provided"
        
        attr_parts = []
        for attr_type, values in attributes.items():
            # Skip visual reasoning (internal model thought process)
            if attr_type == "visual_reasoning":
                continue
                
            if values:
                values_str = ", ".join(values)
                attr_parts.append(f"{attr_type}: {values_str}")
        
        return "; ".join(attr_parts) if attr_parts else "no specific attributes provided"
    
    def _create_prompt(self, label: str, attributes: Dict[str, List[str]], is_group: bool = False, member_count: int = 1) -> str:
        """
        Create prompt for object description generation.
        
        The prompt asks the LLM to:
        1. Look at the object(s) in the image
        2. Consider the provided attributes
        3. Generate a natural description that incorporates these attributes
        4. Refine based on what it actually sees (e.g., reflections, textures)
        5. For groups, describe it as multiple objects
        """
        attr_str = self._format_attributes(attributes)
        
        # Qwen-VL requires specific format with image placeholder
        placeholder = "<|image_pad|>"
        
        if is_group and member_count > 1:
            # Pluralize label for groups
            plural_label = f"{label}s" if not label.endswith('s') else label
            
            prompt = (
                f"<|im_start|>system\nYou are a helpful assistant that generates natural language descriptions.\<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"The image shows a group of {member_count} {plural_label}. "
                f"Provided attributes: {attr_str}. "
                f"Generate a concise, natural description (1-2 sentences max) that describes this as MULTIPLE objects, incorporating the provided attributes. "
                f"IMPORTANT: Describe ONLY the visual content. Do NOT mention image artifacts like 'cropped', 'pixellated', 'low resolution', or 'background'. "
                f"Start with 'a group of {member_count}' or '{member_count}' (NOT capitalized). Do NOT end with punctuation. "
                f"Example format: 'a group of 3 people wearing different colored shirts'<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"<|im_start|>system\nYou are a helpful assistant that generates natural language descriptions.\<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"The image shows {label} object(s). "
                f"Provided attributes: {attr_str}. "
                f"Generate a concise, natural description (1-2 sentences max) that incorporates ALL the provided attributes. "
                f"IMPORTANT: Describe ONLY the visual content. Do NOT mention image artifacts like 'cropped', 'pixellated', 'low resolution', or 'background'. "
                f"Start with lowercase 'a' or 'an' (NOT capitalized). Do NOT end with punctuation. "
                f"Example format: 'a large black TV with red reflections'<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        
        return prompt
    
    def generate_object_description(
        self, 
        label: str, 
        attributes: Dict[str, List[str]], 
        object_image: Image.Image,
        is_group: bool = False,
        member_count: int = 1
    ) -> str:
        """
        Generate enhanced description for a single object using VLLM.
        
        Args:
            label: Object class name
            attributes: Dict of attributes
            object_image: PIL Image of the object
            is_group: Whether this is a grouped object
            member_count: Number of members in the group
            
        Returns:
            Enhanced description string
        """
        prompt = self._create_prompt(label, attributes, is_group, member_count)
        
        # VLLM generate() API with multi_modal_data (same as attribute generation)
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"image": object_image}
        }]
        
        # Generate with VLLM
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        
        # Extract generated text
        description = outputs[0].outputs[0].text.strip()
        
        # Remove any trailing punctuation that might have been generated
        description = description.rstrip('.!?,;:')
        
        return description
    
    def batch_generate_object_descriptions(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate descriptions for a batch of objects using VLLM batch inference.
        
        Args:
            batch_data: List of dicts with 'label', 'attributes', 'object_image' keys
            
        Returns:
            List of description strings
        """
        if not batch_data:
            return []
        
        # Prepare batch inputs (same format as attribute generation)
        batch_inputs = []
        for data in batch_data:
            is_group = data.get('is_group', False)
            member_count = data.get('member_count', 1)
            prompt = self._create_prompt(data['label'], data['attributes'], is_group, member_count)
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": data['object_image']}
            })
        
        # Batch generation with VLLM
        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
        
        # Extract descriptions
        descriptions = []
        for output in outputs:
            description = output.outputs[0].text.strip()
            # Remove any trailing punctuation that might have been generated
            description = description.rstrip('.!?,;:')
            descriptions.append(description)
        
        return descriptions
