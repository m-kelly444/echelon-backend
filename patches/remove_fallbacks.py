#!/usr/bin/env python3
import os
import re

def remove_fallbacks():
    """Remove fallbacks from api_server.py"""
    if not os.path.exists('api_server.py'):
        print("api_server.py not found!")
        return False
        
    # Read the original file
    with open('api_server.py', 'r') as f:
        content = f.read()
    
    # Add a flag indicating real data only
    real_data_flag = """# Real data only mode - no fallbacks or synthetics
print("Echelon is running in REAL DATA ONLY mode - using only genuine threat intelligence data")
"""
    
    # Add after imports
    if 'from tenacity import' in content:
        pattern = r'from tenacity import.*\n'
        match = re.search(pattern, content)
        if match:
            content = content[:match.end()] + real_data_flag + content[match.end():]
    
    # Modify prediction endpoint to only use enhanced engine
    if 'if prediction_engine:' in content:
        pattern = r'                # Use enhanced prediction engine if available\n                if prediction_engine:.*?else:.*?# Prepare response\n                    response = \{.*?\}'
        
        replacement = """                # Use enhanced prediction engine only - no fallbacks
                if prediction_engine:
                    try:
                        # Make enhanced prediction with real data only
                        enhanced_result = prediction_engine.predict([cve_year, base_score, days_since_published], cve_id=cve_id)
                        
                        # The enhanced result already includes all the basic prediction data
                        response = enhanced_result
                    except Exception as e:
                        print(f"Error using enhanced prediction: {e}")
                        self.send_response(503)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "error": "Unable to make prediction with real data",
                            "message": "The system is configured to use only real data, but prediction failed.",
                            "details": str(e),
                            "timestamp": datetime.now().isoformat()
                        }).encode())
                        stats["errors"] += 1
                        return
                else:
                    self.send_response(503)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "Real data prediction engine not available",
                        "message": "The system is configured to use only real data, but the prediction engine is not available.",
                        "timestamp": datetime.now().isoformat()
                    }).encode())
                    stats["errors"] += 1
                    return"""
        
        try:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        except:
            print("Warning: Could not modify prediction endpoint. Pattern might not match.")
    
    # Write the updated content
    with open('api_server.py', 'w') as f:
        f.write(content)
    
    print("Successfully removed fallbacks from api_server.py!")
    return True

if __name__ == "__main__":
    remove_fallbacks()
