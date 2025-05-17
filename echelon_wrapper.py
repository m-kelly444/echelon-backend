                      
import os
import sys
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

ml_forecaster_path = os.path.join(current_dir, 'models', 'ml_attack_forecaster.py')
spec = importlib.util.spec_from_file_location('ml_attack_forecaster', ml_forecaster_path)
ml_attack_forecaster = importlib.util.module_from_spec(spec)
sys.modules['ml_attack_forecaster'] = ml_attack_forecaster
spec.loader.exec_module(ml_attack_forecaster)

sys.modules['models'] = type('', (), {})()
sys.modules['models.ml_attack_forecaster'] = ml_attack_forecaster

api_server_path = os.path.join(current_dir, 'api_server.py')
spec = importlib.util.spec_from_file_location('api_server', api_server_path)
api_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_server)
