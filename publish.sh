#!/usr/bin/env bash

set -e

echo "=================================================="
echo "COOPRECSYS PACKAGE PUBLISHER"
echo "=================================================="

echo "[1/5] Building Docker Image"
docker build -t cooprecsys-builder .

echo "[2/5] Building Package"
docker run --rm \
    -v $(pwd):/workspace \
    cooprecsys-builder

echo "[3/5] Checking Dist Folder"
ls -lah dist/

echo "[4/5] Uploading To PyPI"

docker run --rm \
    -e TWINE_USERNAME=__token__ \
    -e TWINE_PASSWORD=$PYPI_TOKEN \
    -v $(pwd):/workspace \
    cooprecsys-builder \
    bash -c "twine upload dist/*"

echo "[5/5] Publish Completed"