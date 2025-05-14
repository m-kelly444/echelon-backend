#!/bin/bash
set -e

echo "🔨 Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "🌱 Setting up initial database and seed data..."
python seed_data.py

echo "✅ Build completed successfully!"
