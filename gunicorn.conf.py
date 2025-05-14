import os

# Basic server configuration
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 1
timeout = 60  # Shorter timeout to prevent hanging processes

# Critical settings to prevent initialization hanging
preload_app = False  # Disable preloading to avoid initialization freezes
worker_class = "sync"
keepalive = 2
worker_connections = 100

# Logging configuration
loglevel = "info"
accesslog = "-"
errorlog = "-"
capture_output = True

# Process naming
proc_name = 'echelon-ml'

# Disable any potentially blocking operations
forwarded_allow_ips = '*'
