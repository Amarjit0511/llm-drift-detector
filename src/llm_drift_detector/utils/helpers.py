"""
Helper functions for LLM Drift Detector.

This module provides various utility functions used throughout the package
for common tasks like file operations, text processing, and error handling.
"""

import os
import json
import logging
import hashlib
import time
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Tuple
from datetime import datetime, timedelta
import urllib.parse
import importlib.util

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Pathlib Path object for the directory
    """
    path_obj = Path(path)
    os.makedirs(path_obj, exist_ok=True)
    return path_obj

def load_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Optional[Dict[str, Any]]: Loaded JSON data or None if file doesn't exist or has invalid format
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        logger.warning(f"JSON file not found: {path_obj}")
        return None
    
    try:
        with open(path_obj, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON file: {path_obj}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file {path_obj}: {str(e)}")
        return None

def save_json_file(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: JSON indentation level
        
    Returns:
        bool: True if successful, False otherwise
    """
    path_obj = Path(file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(path_obj.parent, exist_ok=True)
    
    try:
        with open(path_obj, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {path_obj}: {str(e)}")
        return False

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length-len(suffix)] + suffix

def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def format_time_delta(delta: timedelta) -> str:
    """
    Format a timedelta into a human-readable string.
    
    Args:
        delta: Timedelta to format
        
    Returns:
        str: Formatted time string (e.g., "2 days 3 hours 45 minutes")
    """
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds and not days and not hours:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    
    if not parts:
        return "0 seconds"
    
    return " ".join(parts)

def retry_with_backoff(
    func: Callable[..., R],
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Exception, ...] = (Exception,),
    logger_obj: Optional[logging.Logger] = None
) -> Callable[..., R]:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to multiply backoff time by after each retry
        exceptions: Exceptions to catch and retry on
        logger_obj: Logger to use (defaults to module logger)
        
    Returns:
        Callable: Wrapped function with retry logic
    """
    log = logger_obj or logger
    
    def wrapper(*args, **kwargs):
        retries = 0
        backoff = initial_backoff
        
        while True:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                retries += 1
                if retries > max_retries:
                    log.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                    raise
                
                log.warning(f"Retry {retries}/{max_retries} for {func.__name__} after error: {str(e)}")
                
                # Wait with exponential backoff
                time.sleep(backoff)
                backoff *= backoff_factor
    
    return wrapper

def hash_string(s: str, algorithm: str = "md5") -> str:
    """
    Create a hash of a string.
    
    Args:
        s: String to hash
        algorithm: Hash algorithm to use
        
    Returns:
        str: Hex digest of the hash
    """
    if algorithm == "md5":
        return hashlib.md5(s.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(s.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(s.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def safe_sample_list(items: List[T], k: int) -> List[T]:
    """
    Safely sample k items from a list, handling cases where k > len(items).
    
    Args:
        items: List to sample from
        k: Number of items to sample
        
    Returns:
        List[T]: Sampled items
    """
    if not items:
        return []
    
    if k >= len(items):
        return items.copy()
    
    return random.sample(items, k)

def validate_api_key(api_key: Optional[str], key_type: str = "API") -> bool:
    """
    Validate that an API key is present and looks valid.
    
    Args:
        api_key: API key to validate
        key_type: Type of key for error messages
        
    Returns:
        bool: True if key appears valid, False otherwise
    """
    if not api_key:
        logger.warning(f"Missing {key_type} key")
        return False
    
    # Check minimum length
    if len(api_key) < 8:
        logger.warning(f"{key_type} key too short, likely invalid")
        return False
    
    return True

def parse_timestamp(timestamp: Union[str, datetime, int, float]) -> Optional[datetime]:
    """
    Parse a timestamp from different formats.
    
    Args:
        timestamp: Timestamp in string, datetime, or unix timestamp format
        
    Returns:
        Optional[datetime]: Parsed datetime or None if parsing fails
    """
    if isinstance(timestamp, datetime):
        return timestamp
    
    try:
        if isinstance(timestamp, (int, float)):
            # Unix timestamp (seconds since epoch)
            return datetime.fromtimestamp(timestamp)
        
        # Try parsing ISO format
        return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except:
        try:
            # Try common formats
            import dateutil.parser
            return dateutil.parser.parse(timestamp)
        except:
            logger.warning(f"Could not parse timestamp: {timestamp}")
            return None

def get_nested_dict_value(
    d: Dict[str, Any],
    key_path: str,
    default: Any = None,
    separator: str = "."
) -> Any:
    """
    Get a value from a nested dictionary using a dot notation path.
    
    Args:
        d: Dictionary to get value from
        key_path: Path to the value (e.g., "parent.child.grandchild")
        default: Default value if path not found
        separator: Separator for keys in the path
        
    Returns:
        Any: Value at the path or default if not found
    """
    keys = key_path.split(separator)
    current = d
    
    for key in keys:
        if not isinstance(current, dict):
            return default
        
        if key not in current:
            return default
        
        current = current[key]
    
    return current

def sample_with_distribution(
    population: List[T],
    weights: Optional[List[float]] = None,
    k: int = 1
) -> List[T]:
    """
    Sample from a population with a given weight distribution.
    
    Args:
        population: Items to sample from
        weights: Optional weights for each item
        k: Number of items to sample
        
    Returns:
        List[T]: Sampled items
    """
    if not population:
        return []
    
    if k <= 0:
        return []
    
    if k >= len(population):
        return population.copy()
    
    if weights is None:
        return random.sample(population, k)
    
    if len(weights) != len(population):
        # Fall back to uniform weights if lengths don't match
        return random.sample(population, k)
    
    # Normalize weights
    total = sum(weights)
    if total == 0:
        return random.sample(population, k)
    
    normalized_weights = [w / total for w in weights]
    
    # Sample with replacement
    return random.choices(population, weights=normalized_weights, k=k)

def check_dependencies(dependencies: Dict[str, str]) -> Dict[str, bool]:
    """
    Check if required/optional dependencies are installed.
    
    Args:
        dependencies: Dictionary of dependency names and their import paths
        
    Returns:
        Dict[str, bool]: Dictionary of dependencies and whether they're available
    """
    results = {}
    
    for name, module_path in dependencies.items():
        # Check if module is installed
        is_available = importlib.util.find_spec(module_path) is not None
        results[name] = is_available
        
        if not is_available:
            logger.debug(f"Optional dependency '{name}' is not installed")
    
    return results