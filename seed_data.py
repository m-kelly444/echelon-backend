import os
import json
import hashlib
import random
from datetime import datetime
from echelon.utils.logging import get_logger
from echelon.database import get_db_connection, init_db

logger = get_logger(__name__)

def generate_sample_threats(count=10):
    """This function has been disabled to prevent synthetic data generation"""
    logger.warning("Synthetic threat generation is disabled")
    return []

def store_sample_threats():
    """This function has been disabled to prevent synthetic data storage"""
    logger.warning("Storing sample threats is disabled")
    return 0

if __name__ == "__main__":
    logger.info("Seeding data is disabled - no synthetic data will be generated")
    logger.info("Please implement data importers for real threat intel data")
