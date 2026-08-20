#!/usr/bin/env bash

set -e

echo "========================================="
echo "BUILDING CYTHON MODULES"
echo "========================================="

echo ""
python3 --version
echo ""

rm -rf build/
rm -f CLtowers/*.c
rm -f CLtowers/*.so
echo ""

python3 ./cysetup.py build_ext --inplace

echo "========================================="
echo "BUILD FINISHED"
echo "========================================="
