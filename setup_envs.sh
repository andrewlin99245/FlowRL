I #!/bin/bash

# Installation script for R_server and Flow_Env virtual environments
# Run this script from the project root directory

set -e  # Exit on error

echo "=========================================="
echo "Environment Setup Script"
echo "=========================================="
echo ""

# Check if we're in the correct directory
if [ ! -d "reward-server" ] || [ ! -d "flow_grpo" ]; then
    echo "Error: This script must be run from the project root directory"
    echo "Expected structure:"
    echo "  project_root/"
    echo "    ├── reward-server/"
    echo "    └── flow_grpo/"
    exit 1
fi

# Get Python version
PYTHON_CMD=$(which python3)
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# ==========================================
# Setup R_server environment
# ==========================================
echo "=========================================="
echo "Setting up R_server environment..."
echo "=========================================="

if [ -d "R_server" ]; then
    echo "R_server directory already exists. Skipping creation."
else
    echo "Creating R_server virtual environment..."
    $PYTHON_CMD -m venv R_server
fi

echo "Activating R_server environment..."
source R_server/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f "R_server/requirements.txt" ]; then
    echo "Installing R_server dependencies..."
    # Install mmdet from local reward-server directory (editable mode)
    echo "Installing mmdet from local reward-server directory..."
    pip install -e reward-server/mmdetection

    # Install other requirements (excluding the mmdet line)
    grep -v "mmdet @" R_server/requirements.txt > R_server/requirements_temp.txt
    pip install -r R_server/requirements_temp.txt
    rm R_server/requirements_temp.txt

    echo "R_server environment setup complete!"
else
    echo "Warning: R_server/requirements.txt not found"
    echo "Please create requirements.txt file first"
fi

deactivate
echo ""

# ==========================================
# Setup Flow_Env environment
# ==========================================
echo "=========================================="
echo "Setting up Flow_Env environment..."
echo "=========================================="

if [ -d "Flow_Env" ]; then
    echo "Flow_Env directory already exists. Skipping creation."
else
    echo "Creating Flow_Env virtual environment..."
    $PYTHON_CMD -m venv Flow_Env
fi

echo "Activating Flow_Env environment..."
source Flow_Env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f "Flow_Env/requirements.txt" ]; then
    echo "Installing Flow_Env dependencies..."

    # Install flow_grpo from local directory in editable mode
    echo "Installing flow_grpo from local directory..."
    pip install -e flow_grpo

    # Install other requirements (excluding the flow_grpo git line)
    grep -v "flow_grpo" Flow_Env/requirements.txt > Flow_Env/requirements_temp.txt
    pip install -r Flow_Env/requirements_temp.txt
    rm Flow_Env/requirements_temp.txt

    echo "Flow_Env environment setup complete!"
else
    echo "Warning: Flow_Env/requirements.txt not found"
    echo "Please create requirements.txt file first"
fi

deactivate
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate R_server environment:"
echo "  source R_server/bin/activate"
echo ""
echo "To activate Flow_Env environment:"
echo "  source Flow_Env/bin/activate"
echo ""
echo "Installed packages:"
echo "  R_server: $(R_server/bin/pip list | wc -l) packages"
echo "  Flow_Env: $(Flow_Env/bin/pip list | wc -l) packages"
echo ""
