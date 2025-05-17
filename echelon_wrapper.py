#!/usr/bin/env python3
import os
import sys
import importlib.util

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# First, load the ml_attack_forecaster module directly
ml_forecaster_path = os.path.join(current_dir, 'models', 'ml_attack_forecaster.py')
spec = importlib.util.spec_from_file_location('ml_attack_forecaster', ml_forecaster_path)
ml_attack_forecaster = importlib.util.module_from_spec(spec)
sys.modules['ml_attack_forecaster'] = ml_attack_forecaster
spec.loader.exec_module(ml_attack_forecaster)

# Make the MLAttackForecaster class available at the expected import path
sys.modules['models'] = type('', (), {})()
sys.modules['models.ml_attack_forecaster'] = ml_attack_forecaster

# Now import and execute the api_server module
api_server_path = os.path.join(current_dir, 'api_server.py')
spec = importlib.util.spec_from_file_location('api_server', api_server_path)
api_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_server)
