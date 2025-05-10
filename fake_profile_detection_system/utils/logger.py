import logging
import os

def setup_logger(name='fake_profile_detector', log_file='logs/app.log', level=logging.INFO):
    """Setup logger configuration"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger if not already added
    if not logger.hasHandlers():
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
