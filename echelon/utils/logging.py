import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL, LOG_FILE

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))

# Make sure we don't add handlers multiple times
if not root_logger.handlers:
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler for persistent logs
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10485760, backupCount=5)
    file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    return logger
