import logging
import os
from typing import Dict, Optional


class LoggerFactory:
    """Factory class for creating and managing loggers with consistent configuration."""
    _loggers: Dict[str, logging.Logger] = {}
    _default_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod    
    def get_logger(cls, name: str, level: Optional[int] = None, 
                 format_str: Optional[str] = None) -> logging.Logger:
        """Get or create a configured logger instance."""
        # Return existing logger if already created
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create new logger 
        logger = logging.getLogger(name)
        
        # Set log level from environment variable or parameter or default to INFO
        if level is None:
            env_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
            level = getattr(logging, env_level, logging.INFO)
        
        logger.setLevel(level)
        
        # Create handler and formatter if not already set up
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(format_str or cls._default_format)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        # Store logger in cache
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def set_default_format(cls, format_str: str) -> None:
        """Set the default format string for new loggers."""
        cls._default_format = format_str


# Convenience function 
def get_logger(name: str, level: Optional[int] = None, 
              format_str: Optional[str] = None) -> logging.Logger:
    """Convenience function to get a logger using the factory."""
    return LoggerFactory.get_logger(name, level, format_str)