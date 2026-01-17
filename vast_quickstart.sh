#!/bin/bash
# =============================================================================
# MBA Deobfuscator - Vast.ai Quick Deployment
# Complete setup: clone → install deps → generate test data → ready to train
# =============================================================================

set -e

echo "=========================================="
echo "MBA Deobfuscator - Vast.ai Quick Start"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Clone repo (skip if already in repo directory)
if [ ! -f "setup.sh" ]; then
    echo "Cloning repository..."
    git clone https://github.com/yourusername/mba-deobfuscator.git
    cd mba-deobfuscator
else
    echo -e "${GREEN}Already in repository directory${NC}"
fi

# 2. Setup environment with automatic CUDA detection
echo ""
echo "Setting up Python environment..."
chmod +x setup.sh
./setup.sh

# 3. Activate venv
source .venv/bin/activate

# 4. Create data directory
mkdir -p data

# 5. Generate test dataset (100K samples, ~5 minutes)
echo ""
echo "=========================================="
echo "Generating test dataset (100K samples)"
echo "=========================================="
python scripts/generate_data.py \
    --output data/train_100k.jsonl \
    --samples 100000 \
    --min-depth 1 \
    --max-depth 10

# Split 90/10 train/val
echo "Splitting into train/val..."
head -n 90000 data/train_100k.jsonl > data/train.jsonl
tail -n 10000 data/train_100k.jsonl > data/val.jsonl

echo -e "${GREEN}Created:${NC}"
echo "  data/train.jsonl (90K samples)"
echo "  data/val.jsonl (10K samples)"

# 6. Print next steps
echo ""
echo "=========================================="
echo -e "${GREEN}Setup complete! Ready to train.${NC}"
echo "=========================================="
echo ""
echo "Start training Phase 2 (supervised learning):"
echo "  python scripts/train.py --phase 2 --config configs/phase2.yaml \\"
echo "      --data data/train.jsonl --val-data data/val.jsonl"
echo ""
echo "Or run full 3-phase pipeline:"
echo "  # Phase 1: Contrastive pretraining (40-60h)"
echo "  python scripts/train.py --phase 1 --config configs/phase1.yaml \\"
echo "      --data data/train.jsonl --val-data data/val.jsonl"
echo ""
echo "  # Phase 2: Supervised learning (100-140h, can skip Phase 1)"
echo "  python scripts/train.py --phase 2 --config configs/phase2.yaml \\"
echo "      --data data/train.jsonl --val-data data/val.jsonl"
echo ""
echo "  # Phase 3: RL fine-tuning (20-30h)"
echo "  python scripts/train.py --phase 3 --config configs/phase3.yaml \\"
echo "      --resume checkpoints/phase2_best.pt"
echo ""
echo "Monitor training with TensorBoard:"
echo "  tensorboard --logdir logs/ --port 6006 --bind_all"
echo "  # Access at http://<vast-instance-ip>:6006"
echo ""
echo "For production training (10M samples):"
echo "  python scripts/generate_data.py --output data/train_10M.jsonl \\"
echo "      --samples 10000000 --min-depth 1 --max-depth 14"
echo ""
echo -e "${YELLOW}Note: Test dataset (100K) is for quick validation only.${NC}"
echo -e "${YELLOW}Production training requires 1M-10M samples.${NC}"
echo ""
