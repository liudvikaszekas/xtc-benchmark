# Installation Instructions

## Univlm Repo Setup

The scripts expect the `univlm` repository in the parent directory of the benchmark repo (default):
```
/path/to/
├── xtc-benchmark/scripts/evaluation/univlm_eval/  ← You are here
└── univlm/                      ← Default location
```

If `univlm` is elsewhere, update `UNIVLM_PATH` in both `generate_images_all_models.py` and `answer_image_questions_all_models.py`:
```python
UNIVLM_PATH = Path("/absolute/path/to/univlm")
```

Don't forget to install all submodules of Univlm:

You can run inside the univlm folder:

```python
git submodule update --init --recursive
```
Otherwise, you might encounter the following issue:

```python
Generation completed!
Total models: 6
Successful: 0
✗ MMaDA: No module named 'models'
✗ BLIP3o: No module named 'blip3o'
✗ Show-o2: No module named 'models'
✗ Show-o: No module named 'models'
✗ JanusPro: No module named 'janus'
✗ OmniGen2: No module named 'omnigen2'
```
## Environment Setup

```bash
# Create conda environment
conda env create -f ../environment.yml
conda activate univlm

# Install additional packages
conda install -c nvidia cuda-toolkit
pip install flash-attn tiktoken typeguard
```

## Model Downloads

### BLIP3o Model

1. Download to `~/models/`:
```bash
mkdir -p ~/models && cd ~/models
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BLIP3o/BLIP3o-Model-8B', repo_type='model', local_dir='BLIP3o-Model-8B')"
```

2. Update model path in `generate_images_all_models.py` and `answer_image_questions_all_models.py` if using different path:
```python
"blip3o": {
    "name": "BLIP3o",
    "default_path": "~/models/BLIP3o-Model-8B",  # Update this path if using something different
    "requires_config": False,
},
```

3. **Fix the diffusion decoder config**:

Edit `~/models/BLIP3o-Model-8B/diffusion-decoder/model_index.json`:
- Change `"multimodal_encoder"` to `[null, null]`
- Change `"tokenizer"` to `[null, null]`

Should look like:
```json
{
  "_class_name": "EmuVisualGenerationPipeline",
  "_diffusers_version": "0.21.2",
  ...
    "multimodal_encoder": [
    null,
    null
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "EulerDiscreteScheduler"
  ],
  "tokenizer": [
    null,
    null
  ],
  ...
}
```

### Show-o2 VAE Model

1. Download the VAE:
```bash
cd ~/models
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth
```

2. Update the path in `../configs/showo2_config.yaml`:
```yaml
vae_model:
    type: "wan21"
    pretrained_model_path: "/absolute/path/to/your/models/Wan2.1_VAE.pth"
```
Replace with your actual absolute path (e.g., `/sc/home/username/models/Wan2.1_VAE.pth`)

## Usage

```bash
# Generate images
python call_generate_images.py

# Answer questions about images
python call_answer_questions.py
```

Other models (MMaDA, EMU3, Show-o, Janus Pro, OmniGen2) download automatically on first use.
