#!/usr/bin/env bash
set -e
conda env create -f environment.yml || conda env update -f environment.yml
echo "Run: conda activate unlearning-repo"
