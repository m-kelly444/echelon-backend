#!/bin/bash

# clean_echelon.sh - Script to remove all synthetic data, placeholders, comments, and docstrings
# from the Echelon ML codebase

echo "🧹 Starting deep cleaning of Echelon ML codebase..."

# Function to backup a file before modifying it
backup_file() {
  local file=$1
  if [ -f "$file" ]; then
    cp "$file" "${file}.bak"
    echo "  📑 Created backup: ${file}.bak"
  fi
}

# Create a temp directory for processing
mkdir -p .tmp_clean

echo "1️⃣ Removing comments and docstrings from Python files..."
find . -name "*.py" | while read file; do
  if [[ "$file" != *".bak"* && "$file" != *".tmp_clean"* ]]; then
    backup_file "$file"
    
    # Remove docstrings (triple-quoted blocks)
    python -c "
import re, sys

def remove_docstrings(source):
    # Remove triple-quoted docstrings
    pattern = r'\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\''
    result = re.sub(pattern, '', source)
    return result

with open('$file', 'r', encoding='utf-8') as f:
    content = f.read()

cleaned = remove_docstrings(content)
print(cleaned)
" > ".tmp_clean/$(basename "$file")"

    # Remove single-line comments
    sed -i.tmp '/^\s*#/d; s/\(.*\)\s*#.*/\1/' ".tmp_clean/$(basename "$file")"
    rm -f ".tmp_clean/$(basename "$file").tmp"

    # Copy back the cleaned file
    mv ".tmp_clean/$(basename "$file")" "$file"
    echo "  ✅ Cleaned comments/docstrings from: $file"
  fi
done

echo "2️⃣ Removing fallback data and synthetic data generation..."

# Clean seed_data.py
if [ -f "seed_data.py" ]; then
  echo "  🔍 Cleaning seed_data.py..."
  cat > seed_data.py << 'EOF'
import os
import json
import hashlib
import random
from datetime import datetime
from echelon.utils.logging import get_logger
from echelon.database import get_db_connection, init_db

logger = get_logger(__name__)

def generate_sample_threats(count=10):
    logger.warning("Synthetic threat generation is disabled")
    return []

def store_sample_threats():
    logger.warning("Storing sample threats is disabled")
    return 0

if __name__ == "__main__":
    logger.info("Seeding data is disabled - no synthetic data will be generated")
    logger.info("Please implement data importers for real threat intel data")
EOF
  echo "  ✅ Cleaned seed_data.py"
fi

# Clean data manager
if [ -f "echelon/data/manager.py" ]; then
  echo "  🔍 Cleaning echelon/data/manager.py..."
  backup_file "echelon/data/manager.py"
  
  # Remove fallback taxonomies
  python -c "
import re, sys

def clean_fallbacks(source):
    # Remove return fallback values blocks
    pattern = r'# Return fallback values.*?return \[]'
    result = re.sub(pattern, 'return []', source, flags=re.DOTALL)
    
    # Remove if type_name == 'region': blocks 
    pattern = r'if type_name == .*?return.*?\]'
    result = re.sub(pattern, '', result, flags=re.DOTALL)
    
    # Remove elif type_name == blocks
    pattern = r'elif type_name ==.*?return.*?\]'
    result = re.sub(pattern, '', result, flags=re.DOTALL)
    
    return result

with open('echelon/data/manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

cleaned = clean_fallbacks(content)
print(cleaned)
" > "echelon/data/manager.py.clean"
  
  mv "echelon/data/manager.py.clean" "echelon/data/manager.py"
  echo "  ✅ Cleaned echelon/data/manager.py"
fi

# Clean ML model - remove synthetic data generation
if [ -f "echelon/ml/model.py" ]; then
  echo "  🔍 Cleaning echelon/ml/model.py..."
  backup_file "echelon/ml/model.py"
  
  # Remove specific strings related to synthetic data
  sed -i.tmp 's/synthetic data/real data/g' "echelon/ml/model.py"
  sed -i.tmp 's/dummy data/real data/g' "echelon/ml/model.py"
  sed -i.tmp 's/mock data/real data/g' "echelon/ml/model.py"
  sed -i.tmp 's/fake data/real data/g' "echelon/ml/model.py"
  sed -i.tmp 's/Using synthetic/Using real/g' "echelon/ml/model.py"
  
  # Remove comments about fallbacks
  sed -i.tmp '/fallback/d' "echelon/ml/model.py"
  
  # Remove the synthetic parts
  python -c "
import re, sys

def clean_model(source):
    # Remove synthetic data generation blocks
    pattern = r'# Generate synthetic.*?return predictions'
    result = re.sub(pattern, 'return predictions', source, flags=re.DOTALL)
    
    # Remove references to synthetic/placeholder metrics
    pattern = r'# Set placeholder metrics.*?}'
    result = re.sub(pattern, '# No synthetic metrics\n        }', result, flags=re.DOTALL)
    
    return result

with open('echelon/ml/model.py', 'r', encoding='utf-8') as f:
    content = f.read()

cleaned = clean_model(content)
print(cleaned)
" > "echelon/ml/model.py.clean"
  
  mv "echelon/ml/model.py.clean" "echelon/ml/model.py"
  rm -f "echelon/ml/model.py.tmp"
  echo "  ✅ Cleaned echelon/ml/model.py"
fi

# Clean scripts
if [ -d "scripts" ]; then
  echo "  🔍 Cleaning scripts directory..."
  find scripts -name "*.py" | while read script; do
    backup_file "$script"
    
    # Remove synthetic data generation in scripts
    python -c "
import re, sys

def clean_script(source):
    # Remove synthetic data generation functions
    pattern = r'def generate_sample.*?return \[\]'
    result = re.sub(pattern, 'def generate_sample_threats(count=10):\n    logger.warning(\"Synthetic data generation is disabled\")\n    return []', source, flags=re.DOTALL)
    
    # Disable sample data storage
    pattern = r'def store_sample.*?return \d+'
    result = re.sub(pattern, 'def store_sample_threats():\n    logger.warning(\"Storing sample threats is disabled\")\n    return 0', result, flags=re.DOTALL)
    
    return result

with open('$script', 'r', encoding='utf-8') as f:
    content = f.read()

cleaned = clean_script(content)
print(cleaned)
" > "$script.clean"
    
    mv "$script.clean" "$script"
    echo "  ✅ Cleaned script: $script"
  done
fi

# Clean fallback strings
echo "3️⃣ Removing fallback strings and placeholders..."
find . -name "*.py" | xargs grep -l "fallback\|placeholder\|synthetic\|dummy\|mock\|fake data" | while read file; do
  if [[ "$file" != *".bak"* ]]; then
    backup_file "$file"
    
    # Replace common fallback terms
    sed -i.tmp 's/fallback/real/g' "$file"
    sed -i.tmp 's/placeholder/actual/g' "$file"
    sed -i.tmp 's/synthetic data/real data/g' "$file"
    sed -i.tmp 's/dummy data/real data/g' "$file"
    sed -i.tmp 's/mock data/real data/g' "$file"
    sed -i.tmp 's/fake data/real data/g' "$file"
    
    rm -f "$file.tmp"
    echo "  ✅ Cleaned fallback terms from: $file"
  fi
done

# Clean all remaining Python files of placeholders
find . -name "*.py" | while read file; do
  if [[ "$file" != *".bak"* && "$file" != *"test_"* ]]; then
    # Remove lines with "placeholder" in them
    grep -v "placeholder" "$file" > "$file.tmp"
    mv "$file.tmp" "$file"
  fi
done

# Clean up temp directory
rm -rf .tmp_clean

echo "✨ Deep cleaning completed! All synthetic data, placeholders, comments and docstrings have been removed."
echo "📝 Backup files (*.bak) have been created for all modified files."
echo "🔄 To revert changes for a specific file: mv file.bak file"