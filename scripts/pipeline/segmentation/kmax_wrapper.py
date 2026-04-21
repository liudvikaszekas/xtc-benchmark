"""
Wrapper for kMaX-DeepLab panoptic segmentation model.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.base import BaseSegmentor


class KMaxSegmentor(BaseSegmentor):
    """
    Wrapper for kMaX-DeepLab model.
    
    This assumes kMaX-DeepLab is installed in external/kmax-deeplab/
    and follows the Detectron2-based inference API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize kMaX-DeepLab segmentor.
        
        Args:
            config: Configuration dict with keys:
                - kmax_path: Path to kMaX-DeepLab repository
                - config_file: Path to model config file
                - weights: Path to model weights
                - output_dir: Directory to save visualizations
        """
        super().__init__(config)
        self.kmax_path = Path(config['kmax_path'])
        self.config_file = config['config_file']
        self.weights = config['weights']
        self.output_dir = Path(config.get('output_dir', './data/output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add kMaX-DeepLab to Python path
        if str(self.kmax_path) not in sys.path:
            sys.path.insert(0, str(self.kmax_path))
    
    def load_model(self) -> None:
        """
        Load kMaX-DeepLab model using Detectron2.
        
        This method initializes the model following the kMaX-DeepLab
        inference pattern from their demo/train_net.py
        """
        try:
            # Import detectron2 and kmax components
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.projects.deeplab import add_deeplab_config
            
            # Import kMaX-DeepLab modules to register them
            try:
                # Try standard import first
                from kmax_deeplab import add_kmax_deeplab_config
                import kmax_deeplab  # This registers the models
            except ImportError:
                # Fallback: manually import from kmax_path
                import importlib.util
                
                # Import config
                spec = importlib.util.spec_from_file_location(
                    "kmax_config",
                    self.kmax_path / "kmax_deeplab" / "config.py"
                )
                kmax_config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(kmax_config_module)
                add_kmax_deeplab_config = kmax_config_module.add_kmax_deeplab_config
                
                # Import model to register it
                model_spec = importlib.util.spec_from_file_location(
                    "kmax_model",
                    self.kmax_path / "kmax_deeplab" / "kmax_model.py"
                )
                kmax_model_module = importlib.util.module_from_spec(model_spec)
                spec.loader.exec_module(kmax_model_module)
            
            # Setup config
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_kmax_deeplab_config(cfg)
            cfg.merge_from_file(self.config_file)
            cfg.MODEL.WEIGHTS = self.weights
            cfg.MODEL.DEVICE = 'cuda' if self.config.get('use_cuda', True) else 'cpu'
            
            # Create predictor
            self.model = DefaultPredictor(cfg)
            self.cfg = cfg
            
            print(f"✓ kMaX-DeepLab model loaded successfully")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import kMaX-DeepLab dependencies. "
                f"Make sure detectron2 and kMaX-DeepLab are properly installed.\n"
                f"Error: {str(e)}"
            )
    
    def segment(self, image_path: str) -> Tuple[np.ndarray, List[str], str]:
        """
        Run panoptic segmentation using kMaX-DeepLab.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Tuple of (masks, labels, visualization_path)
        """
        if self.model is None:
            self.load_model()
        
        self.validate_image(image_path)
        
        import cv2
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        
        image = cv2.imread(image_path)
        
        # Run inference
        outputs = self.model(image)
        
        # Extract panoptic segmentation
        panoptic_seg = outputs["panoptic_seg"]
        segments_info = panoptic_seg[1]
        
        panoptic_seg_cpu = panoptic_seg[0].cpu()
        masks = panoptic_seg_cpu.numpy()
        
        # Get metadata for class names
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        
        # Extract labels for each segment
        labels = []
        for segment in segments_info:
            category_id = segment['category_id']
            if category_id < len(metadata.stuff_classes):
                label = metadata.stuff_classes[category_id]
            else:
                label = metadata.thing_classes[category_id - len(metadata.stuff_classes)]
            labels.append(label)
        
        # Create visualization using CPU tensor
        visualizer = Visualizer(image[:, :, ::-1], metadata, scale=1.0)
        vis_output = visualizer.draw_panoptic_seg(panoptic_seg_cpu, segments_info)
        
        # Save visualization
        vis_image = vis_output.get_image()
        output_filename = Path(image_path).stem + "_kmax_seg.jpg"
        vis_path = str(self.output_dir / output_filename)
        cv2.imwrite(vis_path, vis_image[:, :, ::-1])
        
        print(f"✓ Segmentation complete: {len(labels)} segments detected")
        print(f"✓ Visualization saved to: {vis_path}")
        
        return masks, labels, vis_path