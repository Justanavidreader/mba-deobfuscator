# Deployment Guide - Vast.ai & Cloud Platforms

Complete deployment instructions for running MBA Deobfuscator on cloud GPU instances.

**Supported platforms**: Vast.ai, Runpod, Lambda Labs, AWS, GCP, local Linux/WSL

---

## Quick Start (One Command)

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/mba-deobfuscator/master/vast_quickstart.sh | bash
```

Or manually:

```bash
git clone https://github.com/yourusername/mba-deobfuscator.git
cd mba-deobfuscator
bash vast_quickstart.sh
```

This will:
1. Clone repository (if not already cloned)
2. Detect CUDA version and install dependencies
3. Create Python virtual environment
4. Generate 100K sample test dataset
5. Split into train/val (90K/10K)

**Time**: ~15-20 minutes

---

## Step-by-Step Deployment

### 1. Create Cloud Instance

**Recommended Specs**:

| Model Size | GPU | VRAM | RAM | Storage | Estimated Cost/hr (spot) |
|------------|-----|------|-----|---------|--------------------------|
| Base (15M) | RTX 3090/4090 | 24GB | 32GB | 100GB | $0.30-0.50 |
| Scaled (420M) | A100 40GB | 40GB | 64GB | 200GB | $1.50-2.00 |
| Scaled (420M) | 2× RTX 3090 | 2×24GB | 64GB | 200GB | $0.60-1.00 |

**Vast.ai Template Selection**:
- Image: `pytorch/pytorch:2.8.0-cuda12.4-cudnn9-devel`
- Alternative: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- Disk space: 100GB minimum (200GB recommended for production data)

**SSH Access**:
Vast.ai provides SSH command after instance creation:
```bash
ssh -p <port> root@<instance-ip> -L 8080:localhost:8080
```

---

### 2. Clone Repository

```bash
cd /workspace  # Vast.ai default working directory
# Or: cd ~/

git clone https://github.com/yourusername/mba-deobfuscator.git
cd mba-deobfuscator
```

---

### 3. Install Dependencies

The `setup.sh` script automatically detects CUDA and installs all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

**What it does**:
- Detects CUDA version via `nvidia-smi` (fallback to `nvcc`)
- Maps CUDA version to PyTorch wheel (cu118, cu121, cu124, cu126, cu129)
- Creates Python 3.10+ virtual environment at `.venv`
- Installs PyTorch 2.8.0 with correct CUDA support
- Installs PyTorch Geometric + extensions (torch-scatter, torch-sparse, torch-cluster)
- Installs project dependencies (z3-solver, numpy, tensorboard, wandb, pyyaml, tqdm)
- Verifies installation with import checks

**Manual CUDA Override** (if auto-detection fails):
```bash
./setup.sh --cuda cu124    # Force CUDA 12.4
./setup.sh --cuda cu129    # Force CUDA 12.9
./setup.sh --cpu           # CPU-only installation
```

**Custom Virtual Environment Path**:
```bash
./setup.sh --venv /path/to/custom/venv
```

**Skip Virtual Environment** (use system Python):
```bash
./setup.sh --skip-venv
```

**Verify Installation**:
```bash
source .venv/bin/activate
python -c "
import torch
from torch_geometric.data import Data
from src.models.encoder import GATJKNetEncoder
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
print('Installation verified!')
"
```

---

### 4. Generate Training Data

**Test Dataset** (100K samples, ~5 minutes):
```bash
source .venv/bin/activate

python scripts/generate_data.py \
    --output data/train_100k.jsonl \
    --samples 100000 \
    --min-depth 1 \
    --max-depth 10
```

**Production Dataset** (10M samples, ~8-12 hours):
```bash
python scripts/generate_data.py \
    --output data/train_10M.jsonl \
    --samples 10000000 \
    --min-depth 1 \
    --max-depth 14 \
    --num-workers 8  # Parallel generation
```

**Split Train/Val**:
```bash
# 90% train, 10% val
head -n 9000000 data/train_10M.jsonl > data/train.jsonl
tail -n 1000000 data/train_10M.jsonl > data/val.jsonl
```

**Dataset Format** (JSONL):
```json
{"obfuscated": "(x0 & x1) + (x0 ^ x1)", "simplified": "x0 | x1", "depth": 3}
{"obfuscated": "x0 ^ x0", "simplified": "0", "depth": 1}
```

**Alternative: Custom Text Format**:
```bash
# Load from custom format (see src/data/CUSTOM_FORMAT_README.md)
python scripts/test_custom_format_dataset.py \
    --data data/sample_custom_format.txt \
    --batch-size 32
```

---

### 5. Training

**Activate Environment**:
```bash
source .venv/bin/activate
```

**Phase 1: Contrastive Pretraining** (optional, 40-60 hours):
```bash
python scripts/train.py \
    --phase 1 \
    --config configs/phase1.yaml \
    --data data/train.jsonl \
    --val-data data/val.jsonl \
    --checkpoint-dir checkpoints/phase1 \
    --log-dir logs/phase1
```

**Phase 2: Supervised Learning** (main training, 100-140 hours):
```bash
python scripts/train.py \
    --phase 2 \
    --config configs/phase2.yaml \
    --data data/train.jsonl \
    --val-data data/val.jsonl \
    --checkpoint-dir checkpoints/phase2 \
    --log-dir logs/phase2 \
    --resume checkpoints/phase1/phase1_best.pt  # Optional: use pretrained encoder
```

**Phase 3: RL Fine-Tuning** (optional, 20-30 hours):
```bash
python scripts/train.py \
    --phase 3 \
    --config configs/phase3.yaml \
    --resume checkpoints/phase2/phase2_best.pt \
    --checkpoint-dir checkpoints/phase3 \
    --log-dir logs/phase3
```

**Scaled Model (420M params)**:
```bash
python scripts/train.py \
    --phase 2 \
    --config configs/scaled_model.yaml \
    --data data/train.jsonl \
    --val-data data/val.jsonl
```

**Override Config Parameters**:
```bash
python scripts/train.py \
    --phase 2 \
    --config configs/phase2.yaml \
    --batch-size 16 \
    --learning-rate 3e-4 \
    --max-epochs 100
```

---

### 6. Monitoring

**TensorBoard** (real-time metrics):
```bash
tensorboard --logdir logs/ --port 6006 --bind_all
```

Access at `http://<vast-instance-ip>:6006`

**Important**: Open port 6006 in Vast.ai instance settings (Firewall → Add port mapping)

**Weights & Biases** (cloud logging, optional):
```bash
wandb login  # Enter API key from https://wandb.ai/settings
# Training auto-logs if enabled in config (wandb_enabled: true)
```

**Diagnostic Visualizations**:
```bash
# Generate plots from TensorBoard logs
python scripts/visualize_diagnostics.py \
    --tensorboard logs/experiment \
    --output plots/
```

**Check Training Progress**:
```bash
# View recent logs
tail -f logs/phase2/train.log

# Check GPU usage
nvidia-smi -l 1

# Check checkpoint sizes
du -sh checkpoints/
```

---

### 7. Checkpoints & Model Files

**Checkpoint Structure**:
```
checkpoints/
├── phase1_best.pt         # Best Phase 1 encoder (validation loss)
├── phase1_latest.pt       # Latest Phase 1 checkpoint
├── phase2_best.pt         # Best Phase 2 full model
├── phase2_latest.pt       # Latest Phase 2 checkpoint
├── phase3_best.pt         # RL fine-tuned model
└── phase3_latest.pt
```

**Download Checkpoints to Local Machine**:
```bash
# From local machine (not instance)
scp -P <vast-port> root@<vast-ip>:/workspace/mba-deobfuscator/checkpoints/phase2_best.pt .
```

**Checkpoint Contents**:
- Model state dict (encoder + decoder + heads)
- Optimizer state
- Scheduler state
- Training metrics
- Config used for training

**Resume Training from Checkpoint**:
```bash
python scripts/train.py \
    --phase 2 \
    --config configs/phase2.yaml \
    --resume checkpoints/phase2_latest.pt  # Resumes training
```

---

### 8. Evaluation

**Evaluate Trained Model**:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/phase2_best.pt \
    --test-data data/test.jsonl \
    --output results/eval_results.json
```

**Inference (Single Expression)**:
```bash
python scripts/simplify.py \
    --expr "(x0 & x1) + (x0 ^ x1)" \
    --checkpoint checkpoints/phase2_best.pt \
    --mode beam  # or --mode htps
```

**Batch Inference**:
```bash
python scripts/simplify.py \
    --input data/test_expressions.txt \
    --checkpoint checkpoints/phase2_best.pt \
    --output results/simplified.jsonl \
    --batch-size 32
```

---

## Configuration Files

All training configs in `configs/`:

| File | Description | Use Case |
|------|-------------|----------|
| `phase1.yaml` | Contrastive pretraining | Encoder pretraining (optional) |
| `phase2.yaml` | Supervised learning | **Main training** (base model) |
| `phase3.yaml` | RL fine-tuning | PPO with Z3 verification |
| `scaled_model.yaml` | 420M parameter model | Large-scale training (A100) |
| `semantic_hgt.yaml` | Semantic HGT encoder | Property detection research |
| `diagnostics.yaml` | Diagnostic settings | Over-smoothing detection |

**Modify Configs**:
```yaml
# configs/phase2.yaml
training:
  batch_size: 32           # Reduce if OOM
  learning_rate: 1e-4
  max_epochs: 50
  use_amp: true            # Mixed precision (faster, less memory)
  accumulation_steps: 4    # Gradient accumulation for large batches

model:
  encoder_type: gat_jknet  # Or: ggnn, hgt, rgcn, semantic_hgt
  hidden_dim: 256          # 256 (base) or 768 (scaled)
  num_layers: 4            # 4 (base) or 24 (scaled)
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```yaml
   # In config file
   batch_size: 32 → 16 → 8
   ```

2. Enable gradient accumulation:
   ```yaml
   accumulation_steps: 4  # Effective batch = batch_size × accumulation_steps
   ```

3. Enable mixed precision (AMP):
   ```yaml
   use_amp: true
   ```

4. Reduce model size:
   ```yaml
   hidden_dim: 256 → 128
   num_layers: 4 → 3
   ```

5. Limit expression depth during training:
   ```yaml
   max_depth: 14 → 10
   ```

---

### PyG Extension Install Fails

**Symptoms**: `torch_scatter`, `torch_sparse` import errors

**Solution 1 - Manual Install**:
```bash
source .venv/bin/activate

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"  # e.g., "12.4"

# Install extensions with correct CUDA tag
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.8.0+cu124.html
# Replace cu124 with your CUDA version (cu118, cu121, cu126, cu129)
```

**Solution 2 - CPU-Only Fallback**:
```bash
./setup.sh --cpu  # Install CPU-only versions
```

---

### Z3 Solver Timeout

**Symptoms**: Training slows down during Phase 3 (RL)

**Solutions**:
1. Increase timeout:
   ```yaml
   z3_timeout_ms: 1000 → 2000
   ```

2. Disable Z3 during training (only use at eval):
   ```yaml
   use_z3_verification: false
   ```

3. Use execution-only verification (Tier 2):
   ```yaml
   max_verification_tier: execution  # Skip Z3 (Tier 3)
   ```

---

### Data Generation Slow

**Symptoms**: `generate_data.py` takes too long

**Solutions**:
1. Use multiple workers:
   ```bash
   python scripts/generate_data.py --num-workers 8
   ```

2. Pre-generate dataset before cloud training:
   - Generate on local machine
   - Upload to cloud storage (S3, GCS, Dropbox)
   - Download to instance: `wget <url> -O data/train.jsonl`

3. Use smaller test dataset first:
   ```bash
   --samples 100000  # Test with 100K before 10M
   ```

---

### Instance Interruption (Spot Instances)

**Symptoms**: Training stops mid-run (spot instance terminated)

**Solutions**:
1. Enable auto-checkpointing:
   ```yaml
   save_every_n_epochs: 1  # Save every epoch
   ```

2. Resume from latest checkpoint:
   ```bash
   python scripts/train.py --phase 2 --config configs/phase2.yaml \
       --resume checkpoints/phase2_latest.pt
   ```

3. Use on-demand instances for critical runs (more expensive but reliable)

---

## Cost Optimization

### Storage

**Reduce checkpoint storage**:
```yaml
# In config
keep_only_best: true        # Delete non-best checkpoints
save_every_n_epochs: 5      # Save less frequently
```

**Compress datasets**:
```bash
gzip data/train_10M.jsonl  # Compress when not in use
gunzip data/train_10M.jsonl.gz  # Decompress before training
```

**Clean up intermediate files**:
```bash
rm -rf logs/old_experiments
rm -f checkpoints/phase*_epoch*.pt  # Keep only best/latest
```

---

### Compute

**Use spot instances** (60-80% cheaper):
- Enable auto-resume from checkpoints (see above)
- Expected interruptions: ~1-2 per week
- Savings: $0.30/hr → $0.10/hr (RTX 3090)

**Run phases separately**:
- Phase 1 on cheap GPU (RTX 3090): Encoder pretraining
- Phase 2 on A100: Main supervised learning
- Phase 3 on RTX 3090: RL fine-tuning (less compute)

**Mixed precision training**:
```yaml
use_amp: true  # ~40% speedup, 30% memory reduction
```

---

### Estimated Total Costs

**Base Model (15M params, RTX 3090 spot)**:
- Phase 1: 50h × $0.30/hr = $15
- Phase 2: 120h × $0.30/hr = $36
- Phase 3: 25h × $0.30/hr = $7.50
- **Total**: ~$60

**Scaled Model (420M params, A100 40GB spot)**:
- Phase 1: 60h × $1.50/hr = $90
- Phase 2: 140h × $1.50/hr = $210
- Phase 3: 30h × $1.50/hr = $45
- **Total**: ~$345

**Production Run (Phase 2 only, skip pretraining)**:
- Base model: ~$36
- Scaled model: ~$210

---

## Platform-Specific Notes

### Vast.ai
- Default workspace: `/workspace`
- Port forwarding: Add in instance settings (TensorBoard: 6006, Jupyter: 8888)
- SSH tunneling: Automatic via vast.ai SSH command

### Runpod
- Default workspace: `/workspace`
- Web terminal: Built-in (no SSH needed)
- File browser: Download checkpoints via UI

### Lambda Labs
- Default workspace: `/home/ubuntu`
- Persistent storage: Separate volume (mount at `/mnt/data`)
- SSH keys: Upload before instance creation

### AWS/GCP
- Use p3.2xlarge (V100) or p4d.24xlarge (A100)
- Persistent storage: EBS volumes (AWS) or persistent disks (GCP)
- Spot instances: Configure auto-resume scripts

---

## Advanced: Multi-GPU Training

**Data Parallel Training** (multiple GPUs on one instance):

```bash
# Use PyTorch DDP
python scripts/train.py \
    --phase 2 \
    --config configs/phase2.yaml \
    --gpus 0,1,2,3  # Use 4 GPUs
```

**Update config**:
```yaml
training:
  distributed: true
  backend: nccl
  world_size: 4  # Number of GPUs
```

**Effective batch size**: `batch_size × num_gpus × accumulation_steps`

---

## Support & Resources

- **Documentation**: `docs/` directory
- **Issues**: https://github.com/yourusername/mba-deobfuscator/issues
- **Configs**: All YAML files in `configs/` with inline comments
- **Tests**: `pytest tests/ -v` (verify installation)

---

## Quick Reference Commands

```bash
# Setup
git clone <repo> && cd mba-deobfuscator
bash vast_quickstart.sh

# Activate environment
source .venv/bin/activate

# Generate data
python scripts/generate_data.py --output data/train.jsonl --samples 1000000

# Train
python scripts/train.py --phase 2 --config configs/phase2.yaml --data data/train.jsonl

# Monitor
tensorboard --logdir logs/ --port 6006 --bind_all

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/phase2_best.pt --test-data data/test.jsonl

# Inference
python scripts/simplify.py --expr "(x&y)+(x^y)" --checkpoint checkpoints/phase2_best.pt
```

---

**Last Updated**: 2025-01-17
