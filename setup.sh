#!/bin/bash
# =============================================================================
# MBA Deobfuscator Setup Script
# Automatically detects CUDA version and installs appropriate dependencies
# Works on: Vast.ai, Runpod, Lambda Labs, local Linux/WSL
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "MBA Deobfuscator Environment Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Detect CUDA Version
# -----------------------------------------------------------------------------
detect_cuda_version() {
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
        if [ -z "$CUDA_VERSION" ]; then
            # Fallback: try nvcc
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
            fi
        fi
    elif command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    fi

    if [ -z "$CUDA_VERSION" ]; then
        echo -e "${YELLOW}Warning: Could not detect CUDA version, defaulting to CPU${NC}"
        CUDA_VERSION="cpu"
    fi

    echo -e "${GREEN}Detected CUDA version: $CUDA_VERSION${NC}"
}

# -----------------------------------------------------------------------------
# Map CUDA version to PyTorch wheel index
# -----------------------------------------------------------------------------
get_torch_index() {
    local cuda_ver=$1
    local cuda_major=$(echo $cuda_ver | cut -d. -f1)
    local cuda_minor=$(echo $cuda_ver | cut -d. -f2)

    if [ "$cuda_ver" = "cpu" ]; then
        echo "cpu"
        return
    fi

    # Map to available PyTorch CUDA versions
    # PyTorch supports: cu118, cu121, cu124, cu126, cu129
    if [ "$cuda_major" -ge 13 ]; then
        # CUDA 13.x -> use cu129 (backward compatible)
        echo "cu129"
    elif [ "$cuda_major" -eq 12 ]; then
        if [ "$cuda_minor" -ge 9 ]; then
            echo "cu129"
        elif [ "$cuda_minor" -ge 6 ]; then
            echo "cu126"
        elif [ "$cuda_minor" -ge 4 ]; then
            echo "cu124"
        else
            echo "cu121"
        fi
    elif [ "$cuda_major" -eq 11 ]; then
        echo "cu118"
    else
        echo -e "${YELLOW}Warning: CUDA $cuda_ver may not be supported, trying cu118${NC}"
        echo "cu118"
    fi
}

# -----------------------------------------------------------------------------
# Get PyG wheel URL for detected CUDA
# -----------------------------------------------------------------------------
get_pyg_wheel_url() {
    local cuda_tag=$1
    # PyG wheels available for specific torch+cuda combinations
    # Using torch 2.8.0 as it has best PyG support
    echo "https://data.pyg.org/whl/torch-2.8.0+${cuda_tag}.html"
}

# -----------------------------------------------------------------------------
# Create virtual environment
# -----------------------------------------------------------------------------
setup_venv() {
    local venv_path=${1:-.venv}

    if [ -d "$venv_path" ]; then
        echo -e "${YELLOW}Virtual environment already exists at $venv_path${NC}"
        read -p "Delete and recreate? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$venv_path"
        else
            echo "Using existing venv"
            return
        fi
    fi

    echo "Creating virtual environment..."
    python3 -m venv "$venv_path"
    echo -e "${GREEN}Created venv at $venv_path${NC}"
}

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------
install_deps() {
    local venv_path=${1:-.venv}
    local cuda_tag=$2

    # Activate venv
    source "$venv_path/bin/activate"

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Install PyTorch
    echo "Installing PyTorch with $cuda_tag..."
    if [ "$cuda_tag" = "cpu" ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        # Use torch 2.8.0 for best PyG compatibility
        pip install torch==2.8.0+$cuda_tag torchvision==0.23.0+$cuda_tag \
            --index-url https://download.pytorch.org/whl/$cuda_tag
    fi

    # Install PyG and extensions
    echo "Installing PyTorch Geometric..."
    pip install torch-geometric

    if [ "$cuda_tag" != "cpu" ]; then
        local pyg_url=$(get_pyg_wheel_url $cuda_tag)
        echo "Installing PyG extensions from $pyg_url..."
        pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f "$pyg_url" || {
            echo -e "${YELLOW}Warning: Some PyG extensions failed to install${NC}"
            echo "Trying individual installs..."
            pip install torch-scatter -f "$pyg_url" || true
            pip install torch-sparse -f "$pyg_url" || true
        }
    fi

    # Install project and remaining deps
    echo "Installing project dependencies..."
    pip install -e ".[dev]"

    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------
verify_install() {
    local venv_path=${1:-.venv}
    source "$venv_path/bin/activate"

    echo ""
    echo "Verifying installation..."
    python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

try:
    from torch_scatter import scatter_mean
    print('torch_scatter: OK')
except ImportError as e:
    print(f'torch_scatter: MISSING ({e})')

try:
    from torch_geometric.data import Data
    print('torch_geometric: OK')
except ImportError as e:
    print(f'torch_geometric: MISSING ({e})')

try:
    from src.models.encoder import GATJKNetEncoder
    print('Project imports: OK')
except ImportError as e:
    print(f'Project imports: FAILED ({e})')
"
    echo ""
}

# -----------------------------------------------------------------------------
# Print usage
# -----------------------------------------------------------------------------
print_usage() {
    echo ""
    echo -e "${GREEN}Setup complete!${NC}"
    echo ""
    echo "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "Quick start commands:"
    echo "  python scripts/train.py --phase 1 --config configs/phase1.yaml"
    echo "  python scripts/evaluate.py --checkpoint best.pt"
    echo "  python scripts/simplify.py --expr '(x&y)+(x^y)'"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    local venv_path=".venv"
    local skip_venv=false
    local cuda_override=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --venv)
                venv_path="$2"
                shift 2
                ;;
            --cuda)
                cuda_override="$2"
                shift 2
                ;;
            --skip-venv)
                skip_venv=true
                shift
                ;;
            --cpu)
                cuda_override="cpu"
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --venv PATH    Virtual environment path (default: .venv)"
                echo "  --cuda VER     Override CUDA version (e.g., cu124, cu129)"
                echo "  --cpu          Force CPU-only installation"
                echo "  --skip-venv    Skip venv creation (use current environment)"
                echo "  -h, --help     Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Detect or use override CUDA version
    if [ -n "$cuda_override" ]; then
        if [ "$cuda_override" = "cpu" ]; then
            CUDA_VERSION="cpu"
            cuda_tag="cpu"
        else
            cuda_tag="$cuda_override"
        fi
        echo -e "${GREEN}Using CUDA override: $cuda_tag${NC}"
    else
        detect_cuda_version
        cuda_tag=$(get_torch_index "$CUDA_VERSION")
    fi

    echo -e "${GREEN}Will install PyTorch with: $cuda_tag${NC}"
    echo ""

    # Setup venv
    if [ "$skip_venv" = false ]; then
        setup_venv "$venv_path"
    fi

    # Install dependencies
    install_deps "$venv_path" "$cuda_tag"

    # Verify
    verify_install "$venv_path"

    # Print usage
    print_usage
}

main "$@"
