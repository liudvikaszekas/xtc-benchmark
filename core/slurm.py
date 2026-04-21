import subprocess
import os
import shlex
from pathlib import Path
from typing import List, Dict, Optional, Any
from .pipeline import PipelineStep

class SlurmJobManager:
    """
    Handles submission of PipelineSteps as Slurm jobs.
    """
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.job_ids: Dict[str, str] = {}  # step.name -> job_id

    def submit(self, step: PipelineStep, run_dir: Path, external_dependencies: Optional[List[str]] = None) -> Optional[str]:
        """
        Submits a single step to Slurm, respecting dependencies.
        Returns the job ID (or a placeholder if dry run).
        """
        cmd_args = step.get_command(run_dir)
        resources = step.get_resources()
        env_vars = step.get_env_vars()
        
        # Build dependency string
        dependency_ids = list(external_dependencies) if external_dependencies else []
        for dep in step.dependencies:
            if dep.name in self.job_ids:
                dependency_ids.append(self.job_ids[dep.name])
            else:
                print(f"WARNING: Dependency {dep.name} for {step.name} has no job ID (maybe skipped?).")

        dep_str = ""
        if dependency_ids:
            # We use afterok (successful completion) by default
            dep_str = f"--dependency=afterok:{':'.join(dependency_ids)}"

        # Construct sbatch script content or command
        # Ideally, we construct a sbatch command directly to avoid creating temporary files for every step,
        # OR we wrap the command in a simple script. 
        # Using --wrap is easiest for simple commands, but complex environments might need a script.
        # Let's use sbatch with directives passed as CLI args or a generated script.
        
        # We will create a submission script in run_dir/jobs/
        job_script_dir = run_dir / "jobs"
        job_script_dir.mkdir(parents=True, exist_ok=True)
        job_script_path = job_script_dir / f"{step.step_id:02d}_{step.name}.slurm"
        
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        out_log = log_dir / f"{step.step_id:02d}_{step.name}_%j.out"
        err_log = log_dir / f"{step.step_id:02d}_{step.name}_%j.err"

        self._write_slurm_script(
            path=job_script_path,
            step=step,
            cmd_args=cmd_args,
            resources=resources,
            env_vars=env_vars,
            out_log=out_log,
            err_log=err_log
        )
        
        submit_cmd = ["sbatch"]
        if dep_str:
            submit_cmd.append(dep_str)
        submit_cmd.append(str(job_script_path))
        
        if self.dry_run:
            print(f"[DRY RUN] Would submit: {' '.join(submit_cmd)}")
            fake_job_id = f"JOB_{step.step_id}"
            self.job_ids[step.name] = fake_job_id
            return fake_job_id
        
        try:
            print(f"Submitting {step.name}...")
            result = subprocess.run(
                submit_cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            # Output format: "Submitted batch job 123456"
            output = result.stdout.strip()
            job_id = output.split()[-1]
            print(f"  -> {output}")
            self.job_ids[step.name] = job_id
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for {step.name}: {e.stderr}")
            raise e

    def _write_slurm_script(
        self, 
        path: Path, 
        step: PipelineStep, 
        cmd_args: List[str], 
        resources: Dict[str, Any],
        env_vars: Dict[str, str],
        out_log: Path,
        err_log: Path
    ):
        """Generates the content of the Slurm script."""
        with open(path, 'w') as f:
            f.write("#!/bin/bash\n")
            # Slurm directives
            f.write(f"#SBATCH --job-name={step.name}\n")
            f.write(f"#SBATCH --output={out_log}\n")
            f.write(f"#SBATCH --error={err_log}\n")
            
            # Map resources to SBATCH defaults if keys exist
            if 'account' in resources:
                f.write(f"#SBATCH --account={resources['account']}\n")
                if resources['account'] == 'aisc':
                    f.write(f"#SBATCH --qos=aisc\n")
            if 'partition' in resources:
                f.write(f"#SBATCH --partition={resources['partition']}\n")
            # Automatically set nodes to 1 for all jobs
            nodes = resources.get('nodes', 1)
            f.write(f"#SBATCH --nodes={nodes}\n")
            # Determine constraints (architecture + GPU type)
            # Slurm requires OR conditions to be grouped or separated by |
            constraints_list = []
            
            # Check for GPU type/constraint in resources
            if 'gpus_type' in resources:
                val = resources['gpus_type']
                if isinstance(val, list):
                    constraints_list.append("(" + "|".join(val) + ")")
                else:
                    constraints_list.append(val)
            elif 'constraint' in resources:
                val = resources['constraint']
                if isinstance(val, list):
                    constraints_list.append("(" + "|".join(val) + ")")
                else:
                    constraints_list.append(val)
            
            if constraints_list:
                # Add base arch requirement with AND (&)
                final_constraint = f"ARCH:X86&{'&'.join(constraints_list)}"
                f.write(f"#SBATCH --constraint={final_constraint}\n")
            
            if 'gpus' in resources:
                f.write(f"#SBATCH --gpus={resources['gpus']}\n")
            
            # Generic resource mapping
            if 'mem' in resources:
                f.write(f"#SBATCH --mem={resources['mem']}\n")
            if 'cpus' in resources:
                f.write(f"#SBATCH --cpus-per-task={resources['cpus']}\n")
            if 'time' in resources:
                f.write(f"#SBATCH --time={resources['time']}\n")
                
            f.write("\n")
            f.write("set -e\n")
            f.write("echo \"Starting job on $(hostname) at $(date)\"\n\n")

            # Always run from the root of the repository so that relative "benchmark/..." paths work correctly
            repo_root = Path(__file__).resolve().parent.parent.parent
            f.write(f"cd \"{repo_root}\"\n\n")

            f.write("CACHE_ROOT=\"$PWD/.cache_job_${SLURM_JOB_ID}\"\n")
            f.write("mkdir -p \"$CACHE_ROOT\"\n\n")
            
            f.write("export HF_HOME=\"$PWD/hf_cache\"\n")
            f.write("mkdir -p \"$HF_HOME\"\n\n")

            # Use job specific cache for others (triton, torch inductor, etc) to avoid conflicts
            f.write("export XDG_CACHE_HOME=\"$CACHE_ROOT/.cache\"\n")
            f.write("export TORCHINDUCTOR_CACHE_DIR=\"$CACHE_ROOT/torchinductor\"\n")
            f.write("export TRITON_CACHE_DIR=\"$CACHE_ROOT/triton\"\n")

            # Avoid VLLM / PyTorch multiprocessing issues
            f.write("export VLLM_WORKER_MULTIPROC_METHOD=spawn\n")

            f.write("mkdir -p \"$XDG_CACHE_HOME\"\n")
            f.write("mkdir -p \"$TORCHINDUCTOR_CACHE_DIR\"\n")
            f.write("mkdir -p \"$TRITON_CACHE_DIR\"\n")

            # Environment Setup
            # The config hints at "use_container": true.
            # If use_container is True, we might assume the command itself handles container invocation
            # OR we need to wrap it here.
            # Looking at existing `run_full_workflow.py` -> `workflow_steps`, it seems `run_command` logic adds PYTHONPATH.
            # We should replicate PYTHONPATH setup.
            
            for key, val in env_vars.items():
                f.write(f"export {key}={val}\n")
            
            # Conda activation (if needed) - user config has "conda_env".
            if 'conda_env' in resources:
                if 'conda_init_script' in resources:
                    f.write(f"{resources['conda_init_script']}\n")
                else:
                    f.write(f"source ~/.bashrc\n")  # Default fallback
                    # Fallback conda finding logic
                    f.write('if command -v conda &> /dev/null; then\n')
                    f.write('    eval "$(conda shell.bash hook)"\n')
                    f.write('fi\n')
                
                f.write(f"conda activate {resources['conda_env']}\n\n")

            # Command execution
            full_cmd = shlex.join(cmd_args)
            f.write(f"echo \"Running command: {full_cmd}\"\n")
            f.write(f"{full_cmd}\n")
            f.write("\n")
            f.write("echo \"Job finished at $(date)\"\n")
