"""
Base LLM Client Abstract Class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from PIL import Image


class LLMClient(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate_object_description(
        self, 
        label: str, 
        attributes: Dict[str, List[str]], 
        object_image: Image.Image
    ) -> str:
        """
        Generate enhanced description for a single object.
        
        Args:
            label: Object class name (e.g., "tv", "giraffe")
            attributes: Dict of attributes {"color": ["black", "red"], "size": ["large"]}
            object_image: PIL Image of the masked object region
            
        Returns:
            str: Enhanced description like "a large black TV with red reflections"
        """
        pass
    
    def batch_generate_object_descriptions(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate descriptions for a batch of objects.
        Default implementation calls generate_object_description sequentially.
        Override for true batch processing.
        
        Args:
            batch_data: List of dicts with 'label', 'attributes', 'object_image' keys
            
        Returns:
            List of description strings
        """
        descriptions = []
        for data in batch_data:
            desc = self.generate_object_description(
                data['label'],
                data['attributes'],
                data['object_image']
            )
            descriptions.append(desc)
        return descriptions
