#!/bin/bash

echo "==============================="
echo "🔧 REPAIRING ECHELON STRUCTURE"
echo "==============================="

# 1. Remove nested Git repo if it exists
if [ -d "echelon/.git" ]; then
  echo "[INFO] Removing nested Git repo in ./echelon..."
  rm -rf echelon/.git
fi

# 2. Move contents from ./echelon to root
if [ -d "echelon" ]; then
  echo "[INFO] Moving contents from ./echelon/ to root..."
  mv echelon/* .
  mv echelon/.* . 2>/dev/null || true
  rm -rf echelon
fi

# 3. Reinitialize Git
echo "[INFO] Reinitializing Git repository..."
rm -rf .git
git init
git add .
git commit -m "Clean slate Echelon repo"
git branch -M main
git remote add origin https://github.com/m-kelly444/echelon-backend.git
git push -u origin main

# 4. Install missing Python dependencies
echo "[INFO] Installing required Python packages..."
pip install sse-starlette feedparser --quiet

# 5. Patch backend to use sse_starlette
echo "[INFO] Fixing FastAPI EventSourceResponse import..."
find backend/ -type f -name "*.py" -exec sed -i '' 's/from fastapi.responses import EventSourceResponse/from sse_starlette.sse import EventSourceResponse/g' {} +

# 6. Change Flask default port to 5050
echo "[INFO] Changing Flask port to 5050..."
find . -type f -name "*.py" -exec sed -i '' 's/port=5000/port=5050/g' {} +
find . -type f -name "*.py" -exec sed -i '' 's/port=8080/port=5050/g' {} +

echo "[SUCCESS] Echelon project structure repaired."
echo "==============================================="
echo "✅ You can now run: python backend/main.py or ./run_real_api.sh"
