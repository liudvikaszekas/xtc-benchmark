from ..core.pipeline import PipelineStep
from pathlib import Path
from typing import List, Dict, Any

class ImageGenerationStep(PipelineStep):
    def __init__(self, model_name: str, config: Dict[str, Any], split_index: int = 0, num_splits: int = 1):
        name = f"image_generation_{model_name}"
        if num_splits > 1:
            name += f"_split_{split_index}"
        super().__init__(name, config)
        self.model_name = model_name
        self.split_index = split_index
        self.num_splits = num_splits
    
    def get_command(self, run_dir: Path) -> List[str]:
        script_path = "benchmark/scripts/pipeline/generate_images.py"
        prompt_path = run_dir / "6_prompt_generation/prompts.json"
        out_dir = run_dir / "7_images" / self.model_name
        
        cmd = ["python", script_path, 
                "--prompts-json", str(prompt_path),
                "--output-dir", str(out_dir),
                "--models", self.model_name]
        
        if self.num_splits > 1:
            cmd.extend(["--split-index", str(self.split_index)])
            cmd.extend(["--num-splits", str(self.num_splits)])
            
        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()


