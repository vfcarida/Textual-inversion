"""
Structured logger utility.
"""
import logging
import sys


def get_logger(name: str = "textual_inversion", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured structured logger.
    
    Args:
        name (str): Name of the logger.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
