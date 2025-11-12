# utils/logger.py
import logging
import os
from datetime import datetime

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure logger
log_filename = f"logs/quantdev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("QuantDev")

# Example usage:
if __name__ == "__main__":
    logger.info("Logger initialized successfully.")
