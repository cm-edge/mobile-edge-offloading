import os
from enum import Enum

# ===== MQTT Configuration =====
class MQTTConfig:
    """Centralized MQTT configuration"""
    
    # Default broker settings - can be overridden by environment variables
    DEFAULT_BROKER_HOSTS = {
        "LAN": "nano-1.local",  # Default external broker IP 
        "bridge": "mqtt-broker",      # Docker service name in bridge compose
        "external": "cm118.cm.in.tum.de",  # Default external broker IP
    }
    
    # Broker configuration
    BROKER_HOST_DEVICE = os.environ.get("MQTT_BROKER_HOST", DEFAULT_BROKER_HOSTS["external"])
    BROKER_HOST_EDGE = os.environ.get("MQTT_BROKER_HOST", DEFAULT_BROKER_HOSTS["external"])
    BROKER_HOST_ROUTER = os.environ.get("MQTT_BROKER_HOST", DEFAULT_BROKER_HOSTS["external"])
    BROKER_HOST_CLOUD = os.environ.get("MQTT_BROKER_HOST", DEFAULT_BROKER_HOSTS["external"])
    BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
    
    # Topics configuration
    DEVICE_REQUESTS_TOPIC = "device/requests"
    DEVICE_RESPONSES_BASE_TOPIC = "device/responses"
    DEVICE_STATUS_TOPIC = "device/status"
    EDGE_REQUESTS_TOPIC = "edge/requests"
    EDGE_STATUS_TOPIC = "edge/status"
    NODE_STATUS_TOPIC = "node/status"
    CLOUD_REQUESTS_TOPIC = "cloud/requests"
    
    # MQTT client settings
    KEEPALIVE = 60
    MAX_RECONNECT_ATTEMPTS = 50
    RECONNECT_DELAY_INITIAL = 5
    RECONNECT_DELAY_FACTOR = 5
    RECONNECT_DELAY_MAX = 120

    @staticmethod 
    def get_response_topic(client_id: str) -> str:
        """Get a device-specific response topic"""
        return f"{MQTTConfig.DEVICE_RESPONSES_BASE_TOPIC}/{client_id}"
# Client status
class CLIENT_STATUS(Enum):
    """Client status enumeration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    
    def __str__(self):
        return self.value

# Client types
# class NODE_TYPE(str, Enum): TODO
class NODE_TYPE(Enum):
    """Client type enumeration"""
    DEVICE = "device"
    EDGE = "edge"
    CLOUD = "cloud"
    ROUTER = "router"
    
    def __str__(self):
        return self.value

# payload type
class PAYLOAD_TYPE(Enum):
    """Payload type enumeration"""
    IMAGE = "image"
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"

    def __str__(self):
        return self.value
    

# ===== Application Configuration =====
class AppConfig:
    """Application-wide configuration"""
    
    # Version information: Bigversion.Minorversion.monthpath.daypath
    VERSION = "0.1.4.22.1"
    
    # Messaging intervals
    MSG_INTERVAL = int(os.environ.get("MSG_INTERVAL", "10"))  # in seconds


# ===== Router Configuration =====
class RouterConfig:
    """Router-specific configuration"""
    
    # Routing probabilities
    EDGE_PROBABILITY = 1  # 70% to edge, 30% to cloud


# ===== Paths Configuration =====
class PathConfig:
    """Configuration for file paths"""
    
    # Base paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Data paths
    DATA_FOLDER = os.environ.get("DATA_FOLDER", "./device/data/")

# ===== Health Check Configuration =====
class HealthCheckConfig:
    """Configuration for health checks"""
    DEFAULT_HOST: str = "localhost"
    DEFAULT_PORT: int = 1883
    DEFAULT_TIMEOUT: int = 5
    DEFAULT_RETRIES: int = 1
    DEFAULT_RETRY_DELAY: int = 2

# ===== Valid image extension Configuration =====
class ImageConfig:
    """Configuration for image processing"""
    VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}