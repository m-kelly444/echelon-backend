#!/bin/bash

echo "[*] Cleaning unnecessary and backup files from the repo..."

# Define patterns to remove
patterns=(
  "*.log"
  "*.tmp"
  "*.bak"
  "*~"
  ".DS_Store"
  "Thumbs.db"
  "*.swp"
  "*.swo"
  "*.pyc"
  "__pycache__"
  "node_modules"
  "dist"
  "build"
  ".next"
  ".turbo"
  ".cache"
  ".env"
  ".env.local"
  ".env.development"
)

for pattern in "${patterns[@]}"; do
  echo "[*] Removing $pattern files..."
  find . -name "$pattern" -type f -print -delete 2>/dev/null
  find . -name "$pattern" -type d -print -exec rm -rf {} + 2>/dev/null
done

echo "[*] Done. Listing remaining directory structure:"
tree -L 2

echo "[✓] Repository cleaned."
