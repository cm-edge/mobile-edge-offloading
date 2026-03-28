import uuid
import os
from shared.config import NODE_TYPE

SUFFIX_DIGITS = 3

# Simple cache to store client IDs
_client_id_cache = {}

def client_id(type: NODE_TYPE) -> str:
    """
    Returns the client ID for the current environment.
    Returns the same ID for the same client type during program execution.
    """
    # Use environment variable if set
    if "CLIENT_ID" in os.environ:
        return os.getenv("CLIENT_ID")
    
    # Return cached ID if exists
    if type in _client_id_cache:
        return _client_id_cache[type]
    
    # Generate new ID and cache it
    new_id = f"{type}-{str(uuid.uuid4())[:SUFFIX_DIGITS]}"
    _client_id_cache[type] = new_id
    
    return new_id