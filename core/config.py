import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class Config:
    """
    Parses and validates the workflow configuration.
    """
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return json.load(f)
            
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """Returns the configuration specific to a step."""
        cfg = self._data.get(step_name, {}).copy()
        
        # Inject global conda_init_script into slurm block if present
        global_init = self.get_global("conda_init_script")
        if global_init:
            if 'slurm' not in cfg:
                cfg['slurm'] = {}
            cfg['slurm']['conda_init_script'] = global_init
            
        # Inject global env_vars into step config
        global_env_vars = self.get_env_vars()
        if global_env_vars:
            cfg['env_vars'] = {**global_env_vars, **cfg.get('env_vars', {})}
            
        return cfg
        
    def get_global(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_env_vars(self) -> Dict[str, str]:
        """Returns the global environment variables to be injected into jobs."""
        return self.get_global("env_vars", {})

    def get_external_gt_dir(self) -> Optional[Path]:
        path = self._data.get('external_gt_run_dir')
        return Path(path) if path else None

    # Helper to merge step-specific Slurm config
    def get_slurm_config(self, step_name: str) -> Dict[str, Any]:
        """
        Merges global Slurm config (if any) with step-specific Slurm config.
        Priority: Step > Global. 
        Note: The provided example config has 'slurm' inside each step block.
        """
        step_cfg = self.get_step_config(step_name)
        slurm_cfg = step_cfg.get('slurm', {})
        
        # Merge other relevant keys that might be outside 'slurm' block but needed for resource
        if 'conda_env' in step_cfg:
            slurm_cfg['conda_env'] = step_cfg['conda_env']
        
        # Sometimes num_gpus is outside
        if 'num_gpus' in step_cfg:
            slurm_cfg['num_gpus'] = step_cfg['num_gpus']
            
        return slurm_cfg
