#!/usr/bin/env bash
#
# Generate deterministic dependency lock file using uv
# Ensures reproducible builds across environments
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

# Verify uv is installed
if ! command -v uv &> /dev/null; then
    log_error "uv is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

log_info "uv version: $(uv --version)"
log_info "Python version: $(python3 --version)"

# Generate base lock file
log_info "Generating base requirements lock file..."
uv pip compile \
    --python-version 3.11 \
    --all-extras \
    -o requirements.lock \
    pyproject.toml

log_success "Generated requirements.lock"

# Generate dev lock file
log_info "Generating dev requirements lock file..."
uv pip compile \
    --python-version 3.11 \
    --all-extras \
    -o requirements-dev.lock \
    pyproject.toml

log_success "Generated requirements-dev.lock"

# Display lock file summary
log_info "Lock file contents:"
echo ""
echo -e "${BLUE}=== requirements.lock ===${NC}"
head -20 requirements.lock
echo "... ($(wc -l < requirements.lock) total lines)"

echo ""
echo -e "${BLUE}=== requirements-dev.lock ===${NC}"
head -20 requirements-dev.lock
echo "... ($(wc -l < requirements-dev.lock) total lines)"

log_success "Dependency lock files generated successfully"
