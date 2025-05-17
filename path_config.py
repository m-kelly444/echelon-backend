   

import os

ECHELON_BASE_PATH = "/Users/maevekelly/3rpejwepewju\"

DATA_PATH = os.path.join(ECHELON_BASE_PATH, "data")
MODELS_PATH = os.path.join(ECHELON_BASE_PATH, "models")
CONFIG_PATH = os.path.join(ECHELON_BASE_PATH, "config")

import sys
for path in [ECHELON_BASE_PATH, MODELS_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)
