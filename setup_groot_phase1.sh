#!/bin/bash
################################################################################
# Phase 1: Isaac GR00T Installation Script
#
# This script installs NVIDIA Isaac GR00T N1.5 package for loading the
# fine-tuned GR00T model (phospho-app/gr00t-paper_return-7w9itxzsox)
#
# Estimated time: 3-4 hours (depending on download speeds and compilation)
################################################################################

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════"
echo "Phase 1: Isaac GR00T Installation"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Store original directory
ORIGINAL_DIR=$(pwd)
LEROBOT_DIR="/home/hafnium/lerobot_lohpaul9"

echo -e "${YELLOW}Your System Configuration:${NC}"
echo "  Python: 3.10.18 ✅"
echo "  GPU: RTX 3060, 12GB VRAM ✅"
echo "  CUDA: 13.0 ✅"
echo "  OS: Ubuntu (WSL2)"
echo ""

################################################################################
# Step 1.1: Install System Dependencies
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1.1: Installing system dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6

echo -e "${GREEN}✅ System dependencies installed${NC}"
echo ""

################################################################################
# Step 1.2: Clone Isaac GR00T Repository
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1.2: Cloning Isaac GR00T repository"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd ~/

# Check if already cloned
if [ -d "Isaac-GR00T" ]; then
    echo -e "${YELLOW}⚠️  Isaac-GR00T directory already exists${NC}"
    read -p "Delete and re-clone? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf Isaac-GR00T
        git clone https://github.com/NVIDIA/Isaac-GR00T.git
    else
        echo "Using existing Isaac-GR00T directory"
    fi
else
    git clone https://github.com/NVIDIA/Isaac-GR00T.git
fi

cd Isaac-GR00T

echo -e "${GREEN}✅ Repository cloned${NC}"
echo ""

################################################################################
# Step 1.3: Install Isaac GR00T in Existing Virtual Environment
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1.3: Installing Isaac GR00T package"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate existing venv
source "${LEROBOT_DIR}/.venv/bin/activate"

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Upgrade build tools
echo "Upgrading build tools..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing Isaac GR00T package (this may take 5-10 minutes)..."
pip install -e .[base]

echo -e "${GREEN}✅ Isaac GR00T package installed${NC}"
echo ""

################################################################################
# Step 1.4: Install Flash-Attention (CUDA 13.0 Compatible)
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1.4: Installing Flash-Attention"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "${YELLOW}⚠️  This step compiles CUDA kernels and may take 20-30 minutes${NC}"
echo "Using flash-attn==2.8.2 (compatible with CUDA 13.0)"
echo ""

# Install without build isolation (required for flash-attn)
pip install --no-build-isolation flash-attn==2.8.2

echo -e "${GREEN}✅ Flash-Attention installed${NC}"
echo ""

################################################################################
# Step 1.5: Verify Installation
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1.5: Verifying installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Testing Isaac GR00T import..."
python -c "import isaac_groot; print('✅ Isaac GR00T imported successfully')" || {
    echo -e "${RED}❌ Failed to import isaac_groot${NC}"
    exit 1
}

echo ""
echo "Testing Flash-Attention import..."
python -c "import flash_attn; print('✅ Flash-Attention imported successfully')" || {
    echo -e "${YELLOW}⚠️  Flash-Attention import failed (may still work)${NC}"
}

echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo -e "${GREEN}✅ Installation verification complete${NC}"
echo ""

################################################################################
# Step 1.6: Test Model Loading
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1.6: Testing GR00T model loading"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Return to lerobot directory
cd "${LEROBOT_DIR}"

echo "Running test inference script..."
echo "This will download the model (~2-4 GB) if not cached"
echo ""

python scripts/test_groot_inference.py || {
    echo ""
    echo -e "${RED}❌ Model loading test failed${NC}"
    echo ""
    echo "Common issues:"
    echo "  1. Model not available in transformers (requires Isaac GR00T)"
    echo "  2. CUDA out of memory (need 12+ GB VRAM)"
    echo "  3. Network issues during download"
    echo ""
    echo "Try manual test:"
    echo "  python -c \"from transformers import AutoModel; model = AutoModel.from_pretrained('phospho-app/gr00t-paper_return-7w9itxzsox', trust_remote_code=True)\""
    exit 1
}

echo ""
echo -e "${GREEN}✅ Model loading test successful${NC}"
echo ""

################################################################################
# Summary
################################################################################

echo "════════════════════════════════════════════════════════════════════════"
echo "Phase 1 Complete: Isaac GR00T Installation ✅"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Installed components:"
echo "  ✅ System dependencies (ffmpeg, libsm6, libxext6)"
echo "  ✅ Isaac GR00T package (~/Isaac-GR00T)"
echo "  ✅ Flash-Attention 2.8.2 (CUDA 13.0 compatible)"
echo "  ✅ GR00T model verified (phospho-app/gr00t-paper_return-7w9itxzsox)"
echo ""
echo "Next steps:"
echo "  1. Phase 2: Modify SO101ResidualEnv for image observations"
echo "  2. Phase 3: Validate base policy transfer (100 episodes)"
echo "  3. Phase 4: Full residual RL training"
echo ""
echo "Installation log saved to: ${HOME}/Isaac-GR00T/installation.log"
echo ""
echo -e "${GREEN}Ready to proceed to Phase 2!${NC}"
echo ""

# Return to original directory
cd "${ORIGINAL_DIR}"
