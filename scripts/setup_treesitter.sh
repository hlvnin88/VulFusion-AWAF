#!/usr/bin/env bash
set -e

mkdir -p vendor
mkdir -p build

# C grammar
if [ ! -d "vendor/tree-sitter-c" ]; then
  git clone https://github.com/tree-sitter/tree-sitter-c vendor/tree-sitter-c
fi

# C++ grammar
if [ ! -d "vendor/tree-sitter-cpp" ]; then
  git clone https://github.com/tree-sitter/tree-sitter-cpp vendor/tree-sitter-cpp
fi

python -m src.dfg.build_languages \
  --output build/my-languages.so \
  --repos vendor/tree-sitter-c vendor/tree-sitter-cpp

echo "[OK] Built build/my-languages.so"