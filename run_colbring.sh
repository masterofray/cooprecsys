#!/usr/bin/env bash

set -e

echo "========================================="
echo "BUILDING CYTHON MODULES"
echo "========================================="

cd src
python3 --version
python3 ./models/arycolbring/cysetup.py build_ext --inplace

echo "========================================="
echo "BUILD FINISHED"
echo "========================================="

