from ..core.pipeline import PipelineStep
from pathlib import Path
from typing import List, Dict, Any, Optional

class SegmentationStep(PipelineStep):
    def __init__(self, config: Dict[str, Any], is_prediction: bool = False, model_name: Optional[str] = None):
        name = "segmentation"
        if is_prediction and model_name:
            name = f"segmentation_{model_name}"
        super().__init__(name, config)
        self.is_prediction = is_prediction
        self.model_name = model_name
        
    def get_command(self, run_dir: Path) -> List[str]:
        # Refined command structure
        script_path = "benchmark/scripts/pipeline/generate_segmasks_kmax.py" # Local copy
        
        # Get config values
        kmax_config = self.config.get('kmax_config', {})
        
        # Construct PsG meta path
        psg_meta_path = self.config.get("psg_meta_path")
        if not psg_meta_path:
             # Fallback
             psg_meta_path = "benchmark/configs/psg_metadata.json"

        m_name = self.model_name or "default"
        if self.is_prediction:
            img_dir = run_dir / "7_images" / m_name
            out_dir = str(run_dir / "8_segmentation_pt" / m_name)
        else:
            img_dir = self.config.get('img_dir_gt')
            out_dir = str(run_dir / "1_segmentation_gt")

        cmd = ["python", script_path, 
               "--img-dir", str(img_dir),
               "--out-dir", out_dir,
               "--psg-meta", psg_meta_path,
               "--kmax-path", kmax_config.get("path", "benchmark/submodules/kmax-deeplab"),
               "--kmax-config", kmax_config.get("config", "benchmark/scripts/pipeline/segmentation/config/kmax_r50.yaml"),
               "--kmax-weights", kmax_config.get("weights", "benchmark/weights/kmax_r50.pth")
               ]
        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()

class SceneGraphGenerationStep(PipelineStep):
    def __init__(self, config: Dict[str, Any], is_prediction: bool = False, model_name: Optional[str] = None):
        name = "scene_graph_generation"
        if is_prediction and model_name:
            name = f"scene_graph_generation_{model_name}"
        super().__init__(name, config)
        self.is_prediction = is_prediction
        self.model_name = model_name

    def get_command(self, run_dir: Path) -> List[str]:
        script_path = "benchmark/scripts/pipeline/generate_sg.py"
        model_dir = self.config.get("model_dir", "benchmark/weights/models/masks-loc-sem")
        
        psg_meta_path = self.config.get("psg_meta_path")
        if not psg_meta_path:
             psg_meta_path = "benchmark/configs/psg_metadata.json"
             
        m_name = self.model_name or "default"
        if self.is_prediction:
            img_dir = run_dir / "7_images" / m_name
            out_dir = str(run_dir / "9_scene_graphs_pt" / m_name)
            anno_path = run_dir / "8_segmentation_pt" / m_name / "anno.json"
        else:
            img_dir = self.config.get('img_dir_gt')
            out_dir = str(run_dir / "2_scene_graphs_gt")
            # Use existing segmentation from Step 1
            anno_path = run_dir / "1_segmentation_gt" / "anno.json"


        cmd = ["python", script_path, 
               "--img-dir", str(img_dir),
               "--out-dir", out_dir,
               "--psg-meta", psg_meta_path,
               "--skip-segmentation",
               "--anno-path", str(anno_path),
             "--model-dir", str(model_dir)
              ]
        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()

class CleanAndRefineRelationsStep(PipelineStep):
    def __init__(self, config: Dict[str, Any], is_prediction: bool = False, model_name: Optional[str] = None, use_flexible_spatial_prompt: bool = False):
        name = "clean_and_refine_relations"
        if is_prediction and model_name:
            name = f"clean_and_refine_relations_{model_name}"
        super().__init__(name, config)
        self.is_prediction = is_prediction
        self.model_name = model_name
        self.use_flexible_spatial_prompt = use_flexible_spatial_prompt

    def get_command(self, run_dir: Path) -> List[str]:
        script_path = "benchmark/scripts/pipeline/clean_and_refine_relations.py"
        
        m_name = self.model_name or "default"
        if self.is_prediction:
            img_dir = run_dir / "7_images" / m_name
            sg_pkl = run_dir / "9_scene_graphs_pt" / m_name / "scene-graph.pkl"
            out_dir = run_dir / "10_clean_and_refine_pt" / m_name
        else:
            img_dir = self.config.get('img_dir_gt')
            sg_pkl = run_dir / "2_scene_graphs_gt/scene-graph.pkl"
            out_dir = run_dir / "3_clean_and_refine_gt"
        
        cmd = ["python", script_path, 
               "--input", str(sg_pkl),
               "--images", str(img_dir),
               "--output", str(out_dir),
               "--psg-meta", self.config.get('psg_meta_path', 'ERROR_NO_META')]
               
        if self.use_flexible_spatial_prompt:
            cmd.append("--use-flexible-spatial-prompt")
               
        # Add optional args
        if 'model' in self.config:
            cmd.extend(["--model", self.config['model']])
        if 'num_gpus' in self.config:
            num_gpus = self.config['num_gpus']
        elif 'slurm' in self.config and 'gpus' in self.config['slurm']:
            num_gpus = self.config['slurm']['gpus']
        else:
            num_gpus = None

        if num_gpus:
            cmd.extend(["--num-gpus", str(num_gpus)])
            
        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()

class GraphMergingStep(PipelineStep):
    def __init__(self, config: Dict[str, Any], is_prediction: bool = False, model_name: Optional[str] = None):
        name = "graph_merging"
        if is_prediction and model_name:
            name = f"graph_merging_{model_name}"
        super().__init__(name, config)
        self.is_prediction = is_prediction
        self.model_name = model_name

    def get_command(self, run_dir: Path) -> List[str]:
        script_path = "benchmark/scripts/pipeline/run_graph_merge.py"
        
        m_name = self.model_name or "default"
        if self.is_prediction:
            anno_path = run_dir / "8_segmentation_pt" / m_name / "anno.json"
            pkl_path = run_dir / "9_scene_graphs_pt" / m_name / "scene-graph.pkl"
            out_dir = run_dir / "11_graph_merge_pt" / m_name
            clean_rel_dir = run_dir / "10_clean_and_refine_pt" / m_name
        else:
            anno_path = run_dir / "1_segmentation_gt/anno.json"
            pkl_path = run_dir / "2_scene_graphs_gt/scene-graph.pkl"
            out_dir = run_dir / "4_graph_merge_gt"
            clean_rel_dir = run_dir / "3_clean_and_refine_gt"
        
        cmd = ["python", script_path, 
               "--anno-json", str(anno_path),
               "--scene-graph-pkl", str(pkl_path),
               "--out-dir", str(out_dir),
               "--padding", str(self.config.get('padding', 10)),
               "--min-group-size", str(self.config.get('min_group_size', 3)),
               "--clean-relations-dir", str(clean_rel_dir)
        ]
       
        # Add threshold args if in config (otherwise script defaults use defaults)
        if 'threshold' in self.config:
            cmd.extend(["--threshold", str(self.config['threshold'])])
            
        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()

class AttributeGenerationStep(PipelineStep):
    def __init__(self, config: Dict[str, Any], is_prediction: bool = False, model_name: Optional[str] = None):
        name = "attribute_generation"
        if is_prediction and model_name:
            name = f"attribute_generation_{model_name}"
        super().__init__(name, config)
        self.is_prediction = is_prediction
        self.model_name = model_name

    def get_command(self, run_dir: Path) -> List[str]:
        script_path = "benchmark/scripts/pipeline/generate_attributes.py"
        
        mapping_path = self.config.get('mapping_json', "benchmark/configs/updated_category_mapping.json")

        m_name = self.model_name or "default"
        if self.is_prediction:
            img_dir = run_dir / "7_images" / m_name
            anno_json = run_dir / "8_segmentation_pt" / m_name / "anno.json"
            seg_dir = run_dir / "8_segmentation_pt" / m_name
            out_dir = run_dir / "12_attributes_pt" / m_name
            sg_dir = run_dir / "11_graph_merge_pt" / m_name
            
            cmd = ["python", script_path, 
                   "--img-dir", str(img_dir),
                   "--output-dir", str(out_dir),
                   "--anno-json", str(anno_json),
                   "--seg-dir", str(seg_dir),
                   "--mapping-json", mapping_path,
                   "--scene-graphs-dir", str(sg_dir)]

        else:
            # GT
            img_dir = self.config.get('img_dir_gt')
            anno_json = run_dir / "1_segmentation_gt/anno.json"
            seg_dir = run_dir / "1_segmentation_gt"
            out_dir = run_dir / "5_attributes_gt"
            sg_dir = run_dir / "4_graph_merge_gt"

            cmd = ["python", script_path, 
                   "--img-dir", str(img_dir),
                   "--output-dir", str(out_dir),
                   "--anno-json", str(anno_json),
                   "--seg-dir", str(seg_dir),
                   "--mapping-json", mapping_path,
                   "--scene-graphs-dir", str(sg_dir)]
               
        if 'model' in self.config:
            cmd.extend(["--model", self.config['model']])
            
        if self.is_prediction:
            cmd.append("--is-prediction-step")
            
        # Add GPU count
        if 'num_gpus' in self.config:
            num_gpus = self.config['num_gpus']
        elif 'slurm' in self.config and 'gpus' in self.config['slurm']:
            num_gpus = self.config['slurm']['gpus']
        else:
            num_gpus = None

        if num_gpus:
            cmd.extend(["--num-gpus", str(num_gpus)])

        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()

class GraphMatchingStep(PipelineStep):
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        name = "graph_matching"
        if model_name:
            name = f"graph_matching_{model_name}"
        super().__init__(name, config)
        self.model_name = model_name

    def get_command(self, run_dir: Path) -> List[str]:
        script_path = "benchmark/scripts/pipeline/run_graph_matching.py"
        gt_sg_dir = run_dir / "4_graph_merge_gt"
        gt_attr_file = run_dir / "5_attributes_gt" / "attributes.json"
        
        pred_sg_dir = run_dir / f"11_graph_merge_pt/{self.model_name}"
        pred_attr_file = run_dir / f"12_attributes_pt/{self.model_name}/attributes.json"
        
        cmd = ["python", script_path, 
               "--gt-sg-dir", str(gt_sg_dir),
               "--gt-attr-file", str(gt_attr_file),
               "--pred-sg-dir", str(pred_sg_dir),
               "--pred-attr-file", str(pred_attr_file),
               "--out-dir", str(run_dir / "final_graphs_pt" / (self.model_name or "default"))]
        
        if 'model' in self.config:
             cmd.extend(["--model", self.config['model']])
             
        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()

class PromptGenerationStep(PipelineStep):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("prompt_generation", config)

    def get_command(self, run_dir: Path) -> List[str]:
        script_path = "benchmark/scripts/pipeline/generate_prompts.py"
        cmd = ["python", script_path, "--run_dir", str(run_dir)]
        
        if self.config.get('refine_sentences'):
            cmd.append("--refine-sentences")
            
        if self.config.get('refine_objects', True):
            cmd.append("--refine-objects")
            
            img_dir = self.config.get('img_dir_gt')
            if img_dir:
                cmd.extend(["--img-dir", str(img_dir)])
                
            seg_dir = run_dir / "1_segmentation_gt"
            cmd.extend(["--seg-dir", str(seg_dir)])
            
        if self.config.get('refine_sentences') or self.config.get('refine_objects', True):
            # Extract LLM parameters
            llm_cfg = self.config.get('llm', {})
            if 'model' in llm_cfg:
                cmd.extend(["--model", llm_cfg['model']])
            
            # Handle GPU count
            num_gpus = llm_cfg.get('num_gpus')
            if not num_gpus:
                 num_gpus = self.config.get('slurm', {}).get('gpus')
            
            if num_gpus:
                cmd.extend(["--num-gpus", str(num_gpus)])
                
            if 'batch_size' in llm_cfg:
                cmd.extend(["--batch-size", str(llm_cfg['batch_size'])])
            if 'max_tokens' in llm_cfg:
                cmd.extend(["--max-tokens", str(llm_cfg['max_tokens'])])
            if 'temperature' in llm_cfg:
                cmd.extend(["--temperature", str(llm_cfg['temperature'])])
            if 'conda_env' in llm_cfg:
                cmd.extend(["--llm-env", llm_cfg['conda_env']])
                
        return cmd

    def get_resources(self) -> Dict[str, Any]:
        return self.get_slurm_config()
