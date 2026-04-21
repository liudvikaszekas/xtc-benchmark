import abc
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json

class PipelineStep(abc.ABC):
    """
    Abstract base class for a single step in the pipeline.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.dependencies: List['PipelineStep'] = []
        self.step_id: int = -1  # Assigned by pipeline

    def add_dependency(self, step: 'PipelineStep'):
        """Adds a dependency that must complete before this step runs."""
        self.dependencies.append(step)

    @abc.abstractmethod
    def get_command(self, run_dir: Path) -> List[str]:
        """
        Returns the command line arguments to execute this step.
        """
        pass

    @abc.abstractmethod
    def get_resources(self) -> Dict[str, Any]:
        """
        Returns Slurm resource requirements for this step.
        Expected keys: 'account', 'partition', 'nodes', 'gpus_type', 'mem', 'cpus', 'time', etc.
        """
        pass
    
    def get_env_vars(self) -> Dict[str, str]:
        """
        Returns environment variables specific to this step merged with global ones.
        """
        env_vars = self.config.get('env_vars', {})
        return env_vars

    def dry_run(self, run_dir: Path) -> str:
        """Returns a string description of what this step would do."""
        cmd = " ".join(self.get_command(run_dir))
        resources = self.get_resources()
        return f"[{self.name}] Command: {cmd} | Resources: {resources}"

    def get_slurm_config(self) -> Dict[str, Any]:
        """Helper to extract slurm config with defaults."""
        # This assumes self.config has the merged step config including slurm keys
        return self.config.get('slurm', {})


class Pipeline:
    """
    Orchestrates the execution of pipeline steps.
    """
    def __init__(self, run_dir: Path, dry_run: bool = False):
        self.run_dir = run_dir
        self.dry_run_mode = dry_run
        self.steps: List[PipelineStep] = []
        self.step_map: Dict[str, PipelineStep] = {}

    def add_step(self, step: PipelineStep, step_id: int = None):
        """Adds a step to the pipeline."""
        if step.name in self.step_map:
            if self.step_map[step.name] is step:
                return # Idempotent
            raise ValueError(f"Step with name {step.name} already exists.")
        
        step.step_id = step_id if step_id is not None else len(self.steps) + 1
        self.steps.append(step)
        self.step_map[step.name] = step

    def get_step(self, name: str) -> Optional[PipelineStep]:
        return self.step_map.get(name)

    def branch(self, parent_step_name: str, branch_steps: List[PipelineStep], step_id: int = None):
        """
        Creates a branching point where multiple steps depend on one parent.
        """
        parent = self.get_step(parent_step_name)
        if not parent:
            raise ValueError(f"Parent step {parent_step_name} not found.")
        
        for step in branch_steps:
            step.add_dependency(parent)
            self.add_step(step, step_id=step_id)

    def chain(self, steps: List[PipelineStep], start_id: int = None):
        """
        Adds a list of steps sequentially, where each depends on the previous one.
        """
        if not steps:
            return
            
        previous = None
        current_id = start_id
        for step in steps:
            if previous:
                step.add_dependency(previous)
            self.add_step(step, step_id=current_id)
            previous = step
            if current_id is not None:
                current_id += 1
