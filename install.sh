#!/usr/bin/env bash
set -euo pipefail

PACKAGE="tax-advisor"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11
INSTALL_DIR="$HOME/.tax-advisor"
VENV_DIR="$INSTALL_DIR/venv"

echo "=== Tax Advisor Installer ==="
echo ""

# Find a suitable Python interpreter
find_python() {
    # Check common names in order of preference
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD=$(find_python) || {
    echo "ERROR: Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} is required but not found."
    echo ""
    echo "Install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ from one of:"
    echo "  - https://www.python.org/downloads/"
    echo "  - brew install python@3.12    (macOS)"
    echo "  - sudo apt install python3.12 (Ubuntu/Debian)"
    echo "  - sudo dnf install python3.12 (Fedora)"
    exit 1
}

PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Found Python $PYTHON_VERSION ($PYTHON_CMD)"

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "Existing virtual environment found at $VENV_DIR"
    read -rp "Reinstall? (y/N): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Skipping venv creation."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Install the package
echo "Installing $PACKAGE from PyPI ..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install "$PACKAGE"

# Download spaCy model required by Presidio
echo "Downloading spaCy language model ..."
"$VENV_DIR/bin/python" -m spacy download en_core_web_lg --quiet

# Create a wrapper script on PATH
BIN_DIR="$INSTALL_DIR/bin"
mkdir -p "$BIN_DIR"
WRAPPER="$BIN_DIR/tax-advisor"

cat > "$WRAPPER" << 'EOF'
#!/usr/bin/env bash
exec "$HOME/.tax-advisor/venv/bin/tax-advisor" "$@"
EOF
chmod +x "$WRAPPER"

# Suggest PATH addition
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
    echo ""
    echo "Add the following to your shell profile (.bashrc, .zshrc, etc.):"
    echo ""
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    echo ""
fi

echo ""
echo "Installation complete! Run 'tax-advisor' to start."
