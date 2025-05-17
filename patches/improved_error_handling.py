import time
import random
import json
import requests
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def rate_limit(calls_per_second=1):
    min_interval = 1.0 / calls_per_second
    last_call_time = [0.0]                                  

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            elapsed = current_time - last_call_time[0]

            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            result = func(*args, **kwargs)
            last_call_time[0] = time.time()
            return result
        return wrapper
    return decorator

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError, TimeoutError))
)
def make_api_request(url, method='get', **kwargs):
                                                                      
    try:
        if method.lower() == 'get':
            response = requests.get(url, timeout=10, **kwargs)
        elif method.lower() == 'post':
            response = requests.post(url, timeout=10, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
                                                               
            retry_after = int(e.response.headers.get('Retry-After', random.uniform(2, 5)))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
                                                   
            raise
        elif 400 <= e.response.status_code < 500:
                                                                   
            print(f"Client error when accessing {url}: {e}")
            return {"error": "Client error", "details": str(e), "status_code": e.response.status_code}
        else:
                                           
            print(f"Server error when accessing {url}: {e}")
            raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"Connection/timeout error when accessing {url}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error when accessing {url}: {e}")
        return {"error": "Unexpected error", "details": str(e)}

def handle_api_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
                           
            print(f"API Error in {func.__name__}: {str(e)}")

            error_response = {
                "error": True,
                "message": str(e),
                "timestamp": time.time(),
                "endpoint": func.__name__
            }

            status_code = 500
            if isinstance(e, ValueError):
                status_code = 400
            elif isinstance(e, KeyError):
                status_code = 400
            elif isinstance(e, FileNotFoundError):
                status_code = 404

            return json.dumps(error_response), status_code
    return wrapper
