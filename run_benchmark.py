import argparse
from pathlib import Path
from typing import Dict, Any, List
import sys

# Ensure the parent directory is in the PYTHONPATH so 'benchmark' can be imported
curr_dir = Path(__file__).resolve().parent
repo_root = curr_dir.parent
sys.path.insert(0, str(repo_root))

from core.pipeline import Pipeline, PipelineStep
from core.config import Config
from core.slurm import SlurmJobManager

# Import steps
from steps.gt_steps import (
    SegmentationStep, SceneGraphGenerationStep, GraphMergingStep, 
    AttributeGenerationStep, GraphMatchingStep, PromptGenerationStep,
    CleanAndRefineRelationsStep
)
from steps.eval_steps import ImageGenerationStep
from steps.judge_steps import SceneGraphEvalStep, VQAGenerationStep, VQAMergeStep, LLMJudgeStep, QuestionGenerationStep

def parse_steps(steps_str: str) -> List[int]:
    steps = []
    for part in steps_str.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            start, end = map(int, part.split('-'))
            steps.extend(range(start, end + 1))
        else:
            steps.append(int(part))
    return steps

def build_pipeline(config_path: Path, run_dir: Path, dry_run: bool, models_override: list = None) -> Pipeline:
    cfg = Config(config_path)
    pipeline = Pipeline(run_dir, dry_run)
    
    # --- Ground Truth Pipeline ---
    # Extract global/shared config values
    clean_cfg = cfg.get_step_config('clean_and_refine_relations')
    psg_meta_path = clean_cfg.get('psg_meta_path')
    
    # 1. Segmentation
    seg_cfg = cfg.get_step_config('segmentation')
    if psg_meta_path: seg_cfg['psg_meta_path'] = psg_meta_path
    seg = SegmentationStep(seg_cfg)
    
    # 2. Scene Graph
    sg_cfg = cfg.get_step_config('scene_graph_generation')
    if psg_meta_path: sg_cfg['psg_meta_path'] = psg_meta_path
    sg = SceneGraphGenerationStep(sg_cfg)

    # 3. Clean and Refine
    clean = CleanAndRefineRelationsStep(clean_cfg, use_flexible_spatial_prompt=False)
    
    # 4. Graph Merging
    merge = GraphMergingStep(cfg.get_step_config('graph_merging'))
    
    # 5. Attributes (GT)
    attr_gt = AttributeGenerationStep(cfg.get_step_config('attribute_generation'), is_prediction=False)
    
    # 6. Prompt Gen
    prompt_cfg = cfg.get_step_config('prompt_generation')
    if 'img_dir_gt' not in prompt_cfg:
        prompt_cfg['img_dir_gt'] = seg_cfg.get('img_dir_gt')
    prompt = PromptGenerationStep(prompt_cfg)
    
    pipeline.chain([seg, sg, clean, merge, attr_gt, prompt], start_id=1)
    
    # 7. Question Generation
    judge_cfg = cfg.get_step_config('evaluation')
    q_gen = None
    if judge_cfg.get('run_vqa', True):
        q_gen = QuestionGenerationStep(judge_cfg)
        q_gen.add_dependency(prompt)
        pipeline.add_step(q_gen, step_id=7)
    
    # --- Image Generation & Evaluation Pipeline ---
    if models_override is not None:
        models = models_override
    else:
        models = cfg.get_global('models', ["showo2", "januspro"]) 
    
    img_gen_cfg = cfg.get_step_config('image_generation')
    attr_cfg = cfg.get_step_config('attribute_generation')
    match_cfg = cfg.get_step_config('graph_matching')
    
    model_splits = img_gen_cfg.get('model_splits', {})
    
    for model in models:
        num_splits = model_splits.get(model, 1)
        
        img_gen_steps = []
        if num_splits > 1:
            print(f"Creating {num_splits} split jobs for {model} image generation.")
            for i in range(num_splits):
                step = ImageGenerationStep(model, img_gen_cfg, split_index=i, num_splits=num_splits)
                step.add_dependency(prompt)
                pipeline.add_step(step, step_id=8)
                img_gen_steps.append(step)
        else:
            step = ImageGenerationStep(model, img_gen_cfg)
            step.add_dependency(prompt)
            pipeline.add_step(step, step_id=8)
            img_gen_steps.append(step)
        
        # 9. Segmentation
        seg_pt = SegmentationStep(seg_cfg, is_prediction=True, model_name=model)
        for step in img_gen_steps:
             seg_pt.add_dependency(step)
        pipeline.add_step(seg_pt, step_id=9)
        
        # 10. Scene Graph
        sg_pt = SceneGraphGenerationStep(sg_cfg, is_prediction=True, model_name=model)
        sg_pt.add_dependency(seg_pt)
        pipeline.add_step(sg_pt, step_id=10)
        
        # 11. Clean and Refine
        clean_pt = CleanAndRefineRelationsStep(clean_cfg, is_prediction=True, model_name=model, use_flexible_spatial_prompt=True)
        clean_pt.add_dependency(sg_pt)
        pipeline.add_step(clean_pt, step_id=11)
        
        # 12. Graph Merge
        merge_pt = GraphMergingStep(cfg.get_step_config('graph_merging'), is_prediction=True, model_name=model)
        merge_pt.add_dependency(clean_pt)
        pipeline.add_step(merge_pt, step_id=12)
        
        # 13. Attribute Generation
        attr_pt = AttributeGenerationStep(attr_cfg, is_prediction=True, model_name=model)
        attr_pt.add_dependency(merge_pt)
        pipeline.add_step(attr_pt, step_id=13)
        
        # 14. Graph Matching
        match_step = GraphMatchingStep(match_cfg, model_name=model)
        match_step.add_dependency(attr_pt)
        pipeline.add_step(match_step, step_id=14)

        # --- Judgement Pipeline ---
        
        # 15. Scene Graph Evaluation
        if judge_cfg.get('run_sg_judge', True):
            sg_eval = SceneGraphEvalStep(judge_cfg, model_name=model)
            sg_eval.add_dependency(match_step)
            pipeline.add_step(sg_eval, step_id=15)
            
            # 16. LLM Judge for Scene Graphs
            sg_llm_judge = LLMJudgeStep(judge_cfg, model_name=model, eval_type="sg")
            sg_llm_judge.add_dependency(sg_eval)
            pipeline.add_step(sg_llm_judge, step_id=16)

        # 17. VQA Evaluation
        if judge_cfg.get('run_vqa', True):
            # Split VQA to chunks based on config if specified, default 1
            vqa_num_splits = judge_cfg.get('vqa_splits', {}).get(model, 1)
            vqa_steps = []
            
            for i in range(vqa_num_splits):
                vqa_step = VQAGenerationStep(judge_cfg, model_name=model, split_index=i, num_splits=vqa_num_splits)
                for step in img_gen_steps:
                    vqa_step.add_dependency(step)
                if q_gen:
                    vqa_step.add_dependency(q_gen)
                pipeline.add_step(vqa_step, step_id=17)
                vqa_steps.append(vqa_step)
                
            # Merge VQA
            vqa_merge = VQAMergeStep(judge_cfg, model_name=model)
            for v_step in vqa_steps:
                vqa_merge.add_dependency(v_step)
            pipeline.add_step(vqa_merge, step_id=18)
            
            # 19. LLM Judge for VQA
            vqa_llm_judge = LLMJudgeStep(judge_cfg, model_name=model, eval_type="vqa")
            vqa_llm_judge.add_dependency(vqa_merge)
            pipeline.add_step(vqa_llm_judge, step_id=19)

    return pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the VLM Benchmark Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory for this run")
    parser.add_argument("--steps", type=str, default="1-19", help="Comma-separated steps or ranges (e.g., '1-5,7,16-19')")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to run (overrides config). Example: bagel,blip3o,januspro")
    parser.add_argument("--job-dependency", type=str, default="", help="External job ID to wait for before starting the first step")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    
    allowed_steps = parse_steps(args.steps)
    
    # Parse models override if provided
    models_override = None
    if args.models:
        models_override = [m.strip() for m in args.models.split(",")]
        print(f"Running with models override: {models_override}")
    
    pipeline = build_pipeline(config_path, run_dir, args.dry_run, models_override=models_override)
    manager = SlurmJobManager(dry_run=args.dry_run)
    
    print(f"Starting pipeline execution in {run_dir}...")
    print(f"Allowed steps: {allowed_steps}")
    
    # We need to find the minimum allowed step ID to apply external dependency
    min_allowed_step = min(allowed_steps)
    
    for step in pipeline.steps:
        if step.step_id in allowed_steps:
            ext_deps = [args.job_dependency] if args.job_dependency and step.step_id == min_allowed_step else []
            job_id = manager.submit(step, run_dir, external_dependencies=ext_deps)
            if not args.dry_run:
                print(f"Step {step.step_id} [{step.name}]: Submitted Job {job_id}")
        else:
             print(f"Skipping Step {step.step_id} [{step.name}]")

    job_ids_file = run_dir / "jobs_submitted.txt"
    with open(job_ids_file, "w") as f:
        for job_id in manager.job_ids.values():
            if job_id and "JOB_" not in job_id: 
                f.write(f"{job_id}\n")
    print(f"\nJob IDs saved to: {job_ids_file}")

if __name__ == "__main__":
    main()
