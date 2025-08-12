"""
Logging configuration for X Network Visualization.
"""
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name and standard configuration.
    
    Args:
        name: Logger name, typically __name__
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, assume it's configured
    if logger.handlers:
        return logger
    
    # Configure the logger with standard settings
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger 