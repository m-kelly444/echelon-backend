#!/bin/bash
set -e

# Explicitly set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export PYTHONPATH=$(pwd)
export LOG_LEVEL=DEBUG

# Use system Python path
PYTHON_PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

# Ensure DB directory exists
mkdir -p data

# Initialize database with seed data
echo "🌱 Setting up database with sample data..."
$PYTHON_PATH seed_data.py

# Start the application with verbose output
echo "🚀 Starting application..."
$PYTHON_PATH -m gunicorn 'app:create_app()' --bind 0.0.0.0:5000 --workers 1 --timeout 120 --log-level debug
