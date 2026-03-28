#!/usr/bin/env python3
import os
import sys
import socket
import time
import logging
import errno
import argparse
from shared.logger_factory import get_logger
from shared.config import HealthCheckConfig
from contextlib import closing


# Configure basic logging
logger = get_logger('health_check')

def check_mqtt_connectivity(host=None, port=None, retries=1, retry_delay=2):
    """
    Check MQTT broker connectivity by attempting to establish a TCP connection
    
    Args:
        host: MQTT broker hostname/IP (overrides env var)
        port: MQTT broker port (overrides env var)
        retries: Number of connection attempts before failing
        retry_delay: Delay between retries in seconds
    
    Returns:
        bool: True if connection succeeded, False otherwise
    """
    # Use parameters or environment variables with defaults
    broker_host = host or os.environ.get("MQTT_BROKER_HOST", HealthCheckConfig.DEFAULT_HOST)
    broker_port = int(port or os.environ.get("MQTT_BROKER_PORT", HealthCheckConfig.DEFAULT_PORT))
    
    logger.debug(f"Checking MQTT connectivity to {broker_host}:{broker_port}")
    
    for attempt in range(retries):
        try:
            # Create a socket and attempt to connect
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(HealthCheckConfig.DEFAULT_TIMEOUT)  # Set a timeout for the connection attempt

                # Try to connect
                result = s.connect_ex((broker_host, broker_port))
                if result == 0:
                    logger.debug("MQTT connectivity check successful")
                    return True
                else:
                    error_messages = {
                        errno.ECONNREFUSED: "Connection refused",
                        errno.ETIMEDOUT: "Connection timed out",
                        errno.ENETUNREACH: "Network unreachable"
                    }
                error_msg = error_messages.get(result, f"Unknown error (code: {result})")
                logger.warning(f"Failed to connect to MQTT broker at {broker_host}:{broker_port} - {error_msg}")
        
        except socket.gaierror as e:
            logger.warning(f"DNS resolution failed: {str(e)}")
        except Exception as e:
            logger.warning(f"Unexpected connection error: {str(e)}")
            
        # If we get here, connection failed - retry after delay
        if attempt < retries - 1:
            logger.debug(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error(f"All MQTT connectivity checks failed after {retries} attempts")
    return False

if __name__ == "__main__":
    # When run directly, exit with appropriate code for Docker health checks
    # Exit code 0 means healthy, anything else means unhealthy
    if check_mqtt_connectivity(retries=2):
        sys.exit(0)  # Success/healthy
    else:
        sys.exit(1)  # Failure/unhealthy