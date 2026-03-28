import time
from datetime import datetime
import json
import paho.mqtt.client as mqtt

# Import our shared logger factory and config
from shared.logger_factory import get_logger
from shared.config import MQTTConfig, NODE_TYPE, CLIENT_STATUS 


MAX_RECONNECT_ATTEMPTS = MQTTConfig.MAX_RECONNECT_ATTEMPTS
MQTT_BROKER = MQTTConfig.BROKER_HOST_DEVICE
MQTT_PORT = MQTTConfig.BROKER_PORT

# Configure logging using our factory
logger = get_logger('mqtt_utils')


def status_register(client, client_id: str, node_type: NODE_TYPE):
    status_message = {
        "client_id": client_id,
        "timestamp": datetime.now().isoformat(),
        "status": CLIENT_STATUS.CONNECTED,
        "node_type": node_type,
    }
    client.publish(MQTTConfig.NODE_STATUS_TOPIC, json.dumps(status_message), qos=1)

    last_will_msg = json.dumps({
        "client_id": client_id,
        "status": CLIENT_STATUS.DISCONNECTED,
        "node_type": node_type,
    })
    client.will_set(MQTTConfig.NODE_STATUS_TOPIC, last_will_msg, qos=1)


def on_disconnect(client, userdata, rc):
    logger.info("Disconnected from MQTT broker")


def on_publish(client, userdata, mid):
    logger.debug(f"Message {mid} published")


def on_connect(client, userdata, flags, rc):
    CLIENT_ID = userdata['client_id']
    SUBSCRIBE_TOPIC = userdata['subscribe_topic']
    if rc == 0:
        logger.info(f"Connected to MQTT broker with client ID: {CLIENT_ID}")
        client.subscribe(SUBSCRIBE_TOPIC)
        logger.info(f"Subscribed to topic: {SUBSCRIBE_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT broker, return code: {rc}")


def connect_with_retry(client, broker: str, port: int, 
                       keepalive: int = MQTTConfig.KEEPALIVE, 
                       delay_factor: int = MQTTConfig.RECONNECT_DELAY_FACTOR, 
                       loop_start: bool = True) -> bool:
    """
    Connect to MQTT broker with retry logic
    
    Args:
        client: The MQTT client instance to connect
        broker: The MQTT broker hostname or IP
        port: The MQTT broker port
        max_retries: Maximum number of connection attempts
        keepalive: Keepalive interval in seconds
        delay_factor: Multiplier for retry delay calculation
        loop_start: Whether to call client.loop_start() on success
        
    Returns:
        bool: True if connection succeeded, False otherwise
    """
    connected = False
    retry_count = 0
    max_retries = 100
    
    while not connected and retry_count < max_retries:
        try:
            logger.info(f"Connecting to MQTT broker at {broker}:{port}")
            client.connect(broker, port, keepalive=keepalive)
            
            if loop_start:
                client.loop_start()
                
            connected = True
            logger.info("Successfully connected to MQTT broker")
        except Exception as e:
            retry_count += 1
            # Calculate delay with a cap
            delay = min(retry_count * delay_factor, MQTTConfig.RECONNECT_DELAY_MAX)
            logger.warning(f"Connection failed (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("All connection attempts failed.")
    
    return connected

def setup_mqtt_client(client_id: str, subscribe_topic: str, node_type: NODE_TYPE) -> mqtt.Client:
    """Initialize and configure the MQTT client"""
    client = mqtt.Client(client_id, clean_session=True, userdata={'client_id': client_id, 'subscribe_topic': subscribe_topic})
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    client.on_message = lambda c, u, m: NotImplementedError("on_message handler not implemented")

    if not connect_with_retry(client, MQTT_BROKER, MQTT_PORT):
        logger.error("All connection attempts failed. Exiting.")
        return None
    status_register(client, client_id, node_type)
    
    return client