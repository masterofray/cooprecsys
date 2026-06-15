#!/usr/bin/env bash
#
# Production-grade wheel builder with comprehensive validation
# Usage: ./scripts/build-wheel.sh [--test] [--clean]
#

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_WHEEL=false
CLEAN_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_WHEEL=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Clean previous builds
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Cleaning previous builds..."
    rm -rf build/ dist/ src/**/*.so src/**/*.c src/**/*.pyc *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    log_success "Clean complete"
fi

# Verify tools
log_info "Verifying build tools..."
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found"
    exit 1
fi
if ! command -v uv &> /dev/null; then
    log_warn "uv not found, installing via pip"
    pip install uv
fi

log_info "Python: $(python3 --version)"
log_info "uv: $(uv --version)"

# Install build dependencies
log_info "Installing build dependencies..."
uv pip install --system \
    setuptools>=70.0 \
    wheel>=0.42 \
    cython>=3.0 \
    numpy>=1.21 \
    build

# Build wheel
log_info "Building wheel..."
python -m build --wheel --outdir dist/ || {
    log_error "Wheel build failed"
    exit 1
}

WHEEL_FILE=$(ls dist/*.whl | head -1)
WHEEL_NAME=$(basename "$WHEEL_FILE")

log_success "Wheel built: $WHEEL_NAME"
log_info "Location: $(readlink -f "$WHEEL_FILE")"
log_info "Size: $(du -h "$WHEEL_FILE" | cut -f1)"

# Test wheel if requested
if [ "$TEST_WHEEL" = true ]; then
    log_info "Creating test environment..."
    TEST_ENV="test_env_$$"
    uv venv "$TEST_ENV" --python 3.11
    
    log_info "Installing wheel into test environment..."
    source "$TEST_ENV/bin/activate"
    uv pip install "$WHEEL_FILE"
    
    log_info "Verifying wheel installation..."
    python -c "import cooprecsys; print(f'✓ cooprecsys loaded from: {cooprecsys.__file__}')"
    
    log_info "Running tests..."
    uv pip install pytest pytest-cov
    pytest test/ -v --tb=short || {
        log_error "Tests failed"
        deactivate
        exit 1
    }
    
    deactivate
    rm -rf "$TEST_ENV"
    log_success "All tests passed"
fi

log_success "Build process completed successfully"
echo ""
echo "Next steps:"
echo "  1. Verify wheel: unzip -l dist/$WHEEL_NAME"
echo "  2. Install locally: pip install dist/$WHEEL_NAME"
echo "  3. Create release tag: git tag v0.1.0 && git push origin v0.1.0"
