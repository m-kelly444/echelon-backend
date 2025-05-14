#!/bin/bash
set -e

echo "🔨 Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🌱 Creating necessary directories only, NO data initialization..."
mkdir -p data models cache logs

# Create an empty database file if it doesn't exist
touch data/echelon.db

echo "✅ Build completed successfully! Intentionally skipping data initialization."
