#!/usr/bin/env bash

set -e

echo "========================================="
echo "BUILDING CYTHON MODULES"
echo "========================================="

echo ""
python3 --version
echo ""

rm -rf build/
rm -f CLproximity/*.c
rm -f CLproximity/*.so
echo ""

python3 ./cysetup.py build_ext --inplace

echo "========================================="
echo "BUILD FINISHED"
echo "========================================="

