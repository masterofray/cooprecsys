#!/usr/bin/env bash

set -e

echo "========================================="
echo "BUILDING CYTHON MODULES"
echo "========================================="

cd src

python ./models/arycolbring/cysetup.py build_ext --inplace

echo "========================================="
echo "BUILD FINISHED"
echo "========================================="

