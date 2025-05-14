#!/bin/bash
set -e

# Explicitly set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export PYTHONPATH=$(pwd)
export LOG_LEVEL=DEBUG

# Create required directories
mkdir -p data models cache logs

# Initialize database with seed data if it doesn't exist
if [ ! -f "data/echelon.db" ]; then
    echo "🌱 Setting up database with sample data..."
    python seed_data.py
fi

# Start the application with verbose output
echo "🚀 Starting application..."
python -m gunicorn 'app:create_app()' --bind 0.0.0.0:5000 --workers 1 --timeout 120 --log-level debug
