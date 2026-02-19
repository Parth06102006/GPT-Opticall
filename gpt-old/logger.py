import logging
import sys
import os

def create_logger(log_file):
    """Creates and configures a logger for a specific log file."""
    if log_file:
        # Handle log files in current directory
        log_dir = os.path.dirname(log_file)
        if log_dir:  # Only create directory if path is not empty
            os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(log_file)  # Unique name for each logger
    else:
        logger = logging.getLogger(__name__)
        
    logger.setLevel(logging.INFO)
    
    # Clear previous handlers to avoid duplicate logs and file handle leaks
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    # File handler for the current iteration
    file_handler = logging.FileHandler(log_file, mode='a')  # Use append mode
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
