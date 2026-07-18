# Building & Releasing cooprecsys

## Prerequisites
- Python 3.8+
- `uv` (install via: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `build` package (`pip install build`)

## Local Build

### Option 1: Using `uv` directly
```bash
# Install build dependencies
uv pip install setuptools>=70.0 wheel>=0.42 cython>=3.0 numpy>=1.21

# Build wheel
python -m build --wheel --outdir dist/

# Install from wheel
pip install dist/*.whl