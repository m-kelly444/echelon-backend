                      
pass
import os
import re
import sys
import shutil
from datetime import datetime

def backup_file(filename):
                                           
    backup = f"{filename}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(filename, backup)
    print(f"Created backup: {backup}")

def fix_api_server():
                                                                 
    api_file = 'api_server.py'
    
    if not os.path.exists(api_file):
        print(f"Error: {api_file} not found!")
        return False

    backup_file(api_file)

    with open(api_file, 'r') as f:
        content = f.read()

    cves_implemented = "elif path == \"/cves\":" in content
    geo_implemented = "elif path == \"/geo\":" in content
    
    modifications_made = False

    if not cves_implemented or not geo_implemented:
                                                                                     
        not_found_pattern = r'(\s+)(# Not found\s+else:)'
        
        match = re.search(not_found_pattern, content)
        if not match:
            print("Could not find the 'Not found' block in api_server.py!")
            return False
        
        indent = match.group(1)
        not_found_block = match.group(2)

        endpoints_code = []
        
        if not cves_implemented:
            cves_endpoint = f"""{indent}# CVEs endpoint
{indent}elif path == "/cves":
{indent}    self._set_headers()
{indent}    
{indent}    # Get pagination parameters
{indent}    query = urllib.parse.parse_qs(parsed_path.query)
{indent}    limit = min(int(query.get("limit", ["50"])[0]), 500)
{indent}    offset = int(query.get("offset", ["0"])[0])
{indent}    
{indent}    # Load CVEs
{indent}    cves = []
{indent}    cve_dir = "data/processed/cves"
{indent}    if os.path.exists(cve_dir):
{indent}        files = os.listdir(cve_dir)
{indent}        for i, cve_file in enumerate(files[offset:offset+limit]):
{indent}            if cve_file.endswith(".json"):
{indent}                try:
{indent}                    with open(os.path.join(cve_dir, cve_file), "r") as f:
{indent}                        cve = json.load(f)
{indent}                        cves.append(cve)
{indent}                except Exception as e:
{indent}                    print(f"Error loading {{cve_file}}: {{e}}")
{indent}    
{indent}    response = {{
{indent}        "total": len(os.listdir(cve_dir)) if os.path.exists(cve_dir) else 0,
{indent}        "limit": limit,
{indent}        "offset": offset,
{indent}        "cves": cves
{indent}    }}
{indent}    self.wfile.write(json.dumps(response).encode())
pass{indent}# Geographic data endpoint
{indent}elif path == "/geo":
{indent}    self._set_headers()
{indent}    
{indent}    try:
{indent}        # Check if we have processed geographic data
{indent}        if os.path.exists("data/processed/geo/threat_locations.json"):
{indent}            with open("data/processed/geo/threat_locations.json", "r") as f:
{indent}                geo_data = json.load(f)
{indent}        elif os.path.exists("data/processed/geo/abuse_locations.json"):
{indent}            with open("data/processed/geo/abuse_locations.json", "r") as f:
{indent}                geo_data = json.load(f)
{indent}        else:
{indent}            geo_data = []
{indent}        
{indent}        # Return the data
{indent}        response = {{
{indent}            "count": len(geo_data),
{indent}            "locations": geo_data
{indent}        }}
{indent}    except Exception as e:
{indent}        response = {{
{indent}            "error": f"Error loading geographic data: {{str(e)}}",
{indent}            "locations": []
{indent}        }}
{indent}    
{indent}    self.wfile.write(json.dumps(response).encode())
"""
            endpoints_code.append(geo_endpoint)

        combined_endpoints = "\n".join(endpoints_code)
        content = re.sub(not_found_pattern, combined_endpoints + "\n" + match.group(1) + match.group(2), content)
        modifications_made = True

    endpoints_pattern = r'("endpoints": \[\s+)(.+?)(\s+\])'
    endpoints_match = re.search(endpoints_pattern, content, re.DOTALL)
    
    if endpoints_match:
        endpoints_list = endpoints_match.group(2)
        
        cves_in_list = "/cves" in endpoints_list
        geo_in_list = "/geo" in endpoints_list
        
        if not cves_in_list or not geo_in_list:
            new_endpoints = []
            
            if not cves_in_list:
                new_endpoints.append('                    {"path": "/cves", "method": "GET", "description": "List CVEs"}')
            
            if not geo_in_list:
                new_endpoints.append('                    {"path": "/geo", "method": "GET", "description": "Get geographic threat data"}')

            lines = endpoints_list.strip().split('\n')
            last_line = lines[-1]

            if not last_line.rstrip().endswith(','):
                lines[-1] = last_line.rstrip() + ','

            all_endpoints = '\n'.join(lines + new_endpoints)

            content = re.sub(endpoints_pattern, f'\\1{all_endpoints}\\3', content, flags=re.DOTALL)
            modifications_made = True

    if modifications_made:
        with open(api_file, 'w') as f:
            f.write(content)
        print("Successfully fixed missing endpoints in api_server.py!")
        return True
    else:
        print("No modifications needed - endpoints already present.")
        return True

if __name__ == "__main__":
    print("========================================")
    print("ENDPOINT FIXER: Fixing missing endpoints")
    print("========================================")
    
    if fix_api_server():
        print("\nEndpoints fixed successfully! The API server may need to be restarted.")
        print("To restart the server, use Ctrl+C to stop it and then run it again.")
    else:
        print("\nFailed to fix endpoints. Check the error messages above.")
