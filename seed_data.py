from echelon.utils.logging import get_logger
from echelon.database import init_db

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully.")
