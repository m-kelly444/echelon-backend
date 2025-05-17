"""
Path Configuration

This module provides absolute paths to important directories 
in the Echelon system.
"""

import os

# Base path to Echelon system
ECHELON_BASE_PATH = "/Users/maevekelly/3rpejwepewju\"

# Path to data directories
DATA_PATH = os.path.join(ECHELON_BASE_PATH, "data")
MODELS_PATH = os.path.join(ECHELON_BASE_PATH, "models")
CONFIG_PATH = os.path.join(ECHELON_BASE_PATH, "config")

# Add all important paths to Python path
import sys
for path in [ECHELON_BASE_PATH, MODELS_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)
