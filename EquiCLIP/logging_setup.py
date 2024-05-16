import logging
from datetime import datetime
import os

def setup_logging(args):
    # Create a timestamp string for the filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"logs/{timestamp}.log"

    os.makedirs("logs", exist_ok=True)

    # Set up logging configuration
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info(f"cli args: {args}")
