"""
Base interface for panoptic segmentation models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from pathlib import Path
import numpy as np


class BaseSegmentor(ABC):
    """
    Abstract base class for panoptic segmentation models.
    
    All segmentation wrappers (kMaX-DeepLab, FC-CLIP) should inherit from this.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the segmentor.
        
        Args:
            config: Configuration dictionary containing model paths and parameters
        """
        self.config = config
        self.model = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the segmentation model and weights."""
        pass
    
    @abstractmethod
    def segment(self, image_path: str) -> Tuple[np.ndarray, List[str], str]:
        """
        Run panoptic segmentation on an image.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            Tuple containing:
            - masks: numpy array of segmentation masks (H, W) with segment IDs
            - labels: list of class labels for each segment
            - visualization_path: path to the saved visualization with labels
        """
        pass
    
    def validate_image(self, image_path: str) -> None:
        """
        Validate that the image exists and is readable.
        
        Args:
            image_path: Path to check
        
        Raises:
            FileNotFoundError: If image doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            raise ValueError(f"Unsupported image format: {path.suffix}")