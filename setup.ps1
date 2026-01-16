# =============================================================================
# MBA Deobfuscator Setup Script (Windows PowerShell)
# Automatically detects CUDA version and installs appropriate dependencies
# =============================================================================

param(
    [string]$VenvPath = ".venv",
    [string]$CudaOverride = "",
    [switch]$Cpu,
    [switch]$SkipVenv,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-Color {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

function Show-Help {
    Write-Host @"
MBA Deobfuscator Setup Script (Windows)

Usage: .\setup.ps1 [options]

Options:
  -VenvPath PATH    Virtual environment path (default: .venv)
  -CudaOverride VER Override CUDA version (e.g., cu124, cu129)
  -Cpu              Force CPU-only installation
  -SkipVenv         Skip venv creation (use current environment)
  -Help             Show this help

Examples:
  .\setup.ps1                           # Auto-detect CUDA
  .\setup.ps1 -Cpu                      # CPU-only
  .\setup.ps1 -CudaOverride cu129       # Force CUDA 12.9
"@
    exit 0
}

function Get-CudaVersion {
    try {
        $nvidiaSmi = nvidia-smi 2>$null
        if ($nvidiaSmi) {
            $match = $nvidiaSmi | Select-String -Pattern "CUDA Version: (\d+\.\d+)"
            if ($match) {
                return $match.Matches.Groups[1].Value
            }
        }
    } catch {}

    try {
        $nvcc = nvcc --version 2>$null
        if ($nvcc) {
            $match = $nvcc | Select-String -Pattern "release (\d+\.\d+)"
            if ($match) {
                return $match.Matches.Groups[1].Value
            }
        }
    } catch {}

    return "cpu"
}

function Get-TorchCudaTag {
    param([string]$CudaVersion)

    if ($CudaVersion -eq "cpu") { return "cpu" }

    $parts = $CudaVersion -split "\."
    $major = [int]$parts[0]
    $minor = [int]$parts[1]

    if ($major -ge 13) { return "cu129" }
    elseif ($major -eq 12) {
        if ($minor -ge 9) { return "cu129" }
        elseif ($minor -ge 6) { return "cu126" }
        elseif ($minor -ge 4) { return "cu124" }
        else { return "cu121" }
    }
    elseif ($major -eq 11) { return "cu118" }
    else { return "cu118" }
}

function Main {
    if ($Help) { Show-Help }

    Write-Color "==========================================" "Cyan"
    Write-Color "MBA Deobfuscator Environment Setup" "Cyan"
    Write-Color "==========================================" "Cyan"
    Write-Host ""

    # Determine CUDA tag
    if ($Cpu) {
        $cudaTag = "cpu"
        Write-Color "Forcing CPU-only installation" "Yellow"
    } elseif ($CudaOverride) {
        $cudaTag = $CudaOverride
        Write-Color "Using CUDA override: $cudaTag" "Green"
    } else {
        $cudaVersion = Get-CudaVersion
        Write-Color "Detected CUDA version: $cudaVersion" "Green"
        $cudaTag = Get-TorchCudaTag $cudaVersion
    }

    Write-Color "Will install PyTorch with: $cudaTag" "Green"
    Write-Host ""

    # Create venv
    if (-not $SkipVenv) {
        if (Test-Path $VenvPath) {
            $response = Read-Host "Venv exists at $VenvPath. Delete and recreate? [y/N]"
            if ($response -eq "y" -or $response -eq "Y") {
                Remove-Item -Recurse -Force $VenvPath
            }
        }

        if (-not (Test-Path $VenvPath)) {
            Write-Host "Creating virtual environment..."
            python -m venv $VenvPath
            Write-Color "Created venv at $VenvPath" "Green"
        }
    }

    # Activate and install
    $activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    & $activateScript

    Write-Host "Upgrading pip..."
    python -m pip install --upgrade pip

    Write-Host "Installing PyTorch with $cudaTag..."
    if ($cudaTag -eq "cpu") {
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    } else {
        python -m pip install torch==2.8.0+$cudaTag torchvision==0.23.0+$cudaTag `
            --index-url https://download.pytorch.org/whl/$cudaTag
    }

    Write-Host "Installing PyTorch Geometric..."
    python -m pip install torch-geometric

    if ($cudaTag -ne "cpu") {
        $pygUrl = "https://data.pyg.org/whl/torch-2.8.0+$cudaTag.html"
        Write-Host "Installing PyG extensions from $pygUrl..."
        python -m pip install torch-scatter torch-sparse -f $pygUrl
    }

    Write-Host "Installing project dependencies..."
    python -m pip install -e ".[dev]"

    Write-Color "Dependencies installed successfully!" "Green"
    Write-Host ""

    # Verify
    Write-Host "Verifying installation..."
    python -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

try:
    from torch_scatter import scatter_mean
    print('torch_scatter: OK')
except ImportError as e:
    print(f'torch_scatter: MISSING')

try:
    from torch_geometric.data import Data
    print('torch_geometric: OK')
except ImportError as e:
    print(f'torch_geometric: MISSING')

try:
    from src.models.encoder import GATJKNetEncoder
    print('Project imports: OK')
except ImportError as e:
    print(f'Project imports: FAILED ({e})')
"@

    Write-Host ""
    Write-Color "Setup complete!" "Green"
    Write-Host ""
    Write-Host "To activate the environment:"
    Write-Host "  .venv\Scripts\Activate.ps1"
    Write-Host ""
}

Main
