import re
import secrets
import string
from passlib.hash import argon2
import bleach
import sqlparse

def hash_password(password):
    """
    Hash a password using Argon2 (secure password hashing algorithm)
    
    Args:
        password (str): The plain text password
        
    Returns:
        str: Hashed password
    """
    return argon2.hash(password)

def verify_password(stored_hash, provided_password):
    """
    Verify a password against a stored hash
    
    Args:
        stored_hash (str): The stored password hash
        provided_password (str): The plain text password to verify
        
    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        return argon2.verify(provided_password, stored_hash)
    except Exception:
        return False

def generate_secure_token(length=32):
    """
    Generate a cryptographically secure random token
    
    Args:
        length (int): Length of the token
        
    Returns:
        str: Secure random token
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def sanitize_input(input_string):
    """
    Sanitize user input to prevent XSS attacks
    
    Args:
        input_string (str): User input
        
    Returns:
        str: Sanitized input
    """
    if input_string is None:
        return None
    return bleach.clean(str(input_string))

def validate_query_params(params):
    """
    Validate SQL query parameters to prevent injection
    
    Args:
        params (dict): Dictionary of parameter names and values
        
    Returns:
        bool: True if all parameters are valid, False otherwise
    """
    if not params:
        return True
        
    for key, value in params.items():
        # Check for SQL injection patterns
        if isinstance(value, str):
            # Check for common SQL injection patterns
            suspicious_patterns = [
                r'--', r';.*?--', r';.*?#', r'\/\*.*?\*\/', 
                r'UNION\s+ALL\s+SELECT', r'INSERT\s+INTO', r'UPDATE\s+.+?\s+SET',
                r'DELETE\s+FROM', r'DROP\s+TABLE', r'ALTER\s+TABLE'
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return False
    
    return True

def prepare_query(query, params=None):
    """
    Prepare a SQL query with proper parameterization
    
    Args:
        query (str): SQL query with placeholders
        params (dict, optional): Parameters to substitute
        
    Returns:
        tuple: (query, params) ready for execution
    """
    # Parse and validate the query
    parsed = sqlparse.parse(query)
    if not parsed:
        raise ValueError("Invalid SQL query")
    
    # Validate parameters
    if params and not validate_query_params(params):
        raise ValueError("Invalid query parameters detected")
        
    return query, params 