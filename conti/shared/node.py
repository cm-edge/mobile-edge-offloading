import abc
import json
from datetime import datetime
import time
from typing import Any

import paho.mqtt.client as mqtt

from shared.logger_factory import get_logger
from shared.utils import client_id
from shared.config import MQTTConfig, NODE_TYPE, CLIENT_STATUS


class Node(abc.ABC):
    """
    Abstract base class for all node types in the system.
    Provides common functionality for MQTT clients, connection management,
    and message handling.
    """

    def __init__(self, 
                 node_type: NODE_TYPE,
                 broker_host: str,
                 broker_port: int = MQTTConfig.BROKER_PORT,
    ):
        # Generate or get a client ID
        self.node_type : NODE_TYPE = node_type
        self.client_id = client_id(node_type)
        if node_type == NODE_TYPE.DEVICE:
            self.subscribe_topic = f'{MQTTConfig.DEVICE_RESPONSES_BASE_TOPIC}/{self.client_id}'
        elif node_type == NODE_TYPE.ROUTER:
            self.subscribe_topic = MQTTConfig.DEVICE_REQUESTS_TOPIC
        elif node_type == NODE_TYPE.EDGE:
            self.subscribe_topic = MQTTConfig.EDGE_REQUESTS_TOPIC
        elif node_type == NODE_TYPE.CLOUD:
            self.subscribe_topic = MQTTConfig.CLOUD_REQUESTS_TOPIC
        else:
            raise ValueError(f"Invalid node type: {node_type}")
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.logger = get_logger(f"{self.client_id}")
        self.logger.info(f"Client ID: {self.client_id}")
        
        # Initialize MQTT client with clean session for transient connections
        self.connected = False
        self._setup_mqtt_client_with_retry()


    def _setup_mqtt_client_with_retry(self,
            keepalive: int = MQTTConfig.KEEPALIVE, 
            delay_factor: int = MQTTConfig.RECONNECT_DELAY_FACTOR
            ):
        """Set up and configure the MQTT client with connection retry logic"""
        # Create client instance with client ID and user data
        userdata = {
            'client_id': self.client_id, 
            'subscribe_topic': self.subscribe_topic
        }
        try:
            self.client = mqtt.Client(client_id=self.client_id, clean_session=True, userdata=userdata)
            
            # Set up callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_publish = self.on_publish
            self.client.on_message = self.on_message
            
            # Connection retry logic
            connected = False
            retry_count = 0
            max_retries = MQTTConfig.MAX_RECONNECT_ATTEMPTS
            
            while not connected and retry_count < max_retries:
                try:
                    retry_count += 1
                    self.logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}")
                    self.client.connect(self.broker_host, self.broker_port, keepalive=keepalive)
                    self.client.loop_start()
                    self.logger.info("Successfully connected to MQTT broker, loop started")
                    connected = True
                    self.connected = True
                    self._status_register()
                except mqtt.MQTTException as me:
                    self.logger.error(f"Client(ID:{self.client_id}) MQTT protocol error : {me}")
                    self.connected = False
                    return None
                except ConnectionError as e:
                    # Calculate delay with a cap
                    delay = min(retry_count * delay_factor, MQTTConfig.RECONNECT_DELAY_MAX)
                    self.logger.warning(f"Client connection failed (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        self.logger.info(f"Client retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        self.logger.error(f"Client all connection attempts failed.")
                        self.connected = False
                        return None
                except Exception as e:
                    self.logger.error(f"Client unexpected error: {e}")
                    self.connected = False
                    return None

            self.logger.info("Connected to MQTT broker")
            
        except Exception as e:
            self.logger.error(f"Failed to setup MQTT client: {e}")
            self.connected = False

    def _status_register(self):
        self._publish_status(CLIENT_STATUS.CONNECTED)

        last_will_msg = json.dumps({
            "client_id": self.client_id,
            "status": CLIENT_STATUS.DISCONNECTED.value,
            "node_type": self.node_type.value,
        })
        self.client.will_set(MQTTConfig.NODE_STATUS_TOPIC, last_will_msg, qos=1)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.logger.info(f"Connected to MQTT broker with client ID: {self.client_id}")
            client.subscribe(self.subscribe_topic)
            self.logger.info(f"Subscribed to topic: {self.subscribe_topic}")
            self.connected = True
        else:
            self.logger.error(f"Failed to connect to MQTT broker, return code: {rc}")
            self.connected = False

    @abc.abstractmethod
    def on_message(self, client, userdata, msg):
        raise NotImplementedError("on_message method must be implemented in derived classes")

    def on_publish(self, client, userdata, mid):
        self.logger.debug(f"Message {mid} published")

    def on_disconnect(self, client, userdata, rc):
        self.logger.info("Disconnected from MQTT broker")
        self.connected = False

    def _publish_status(self, status: CLIENT_STATUS):
        """Publish client status to the status topic"""
        status_message = {
            "client_id": self.client_id,
            "timestamp": datetime.now().isoformat(),
            "status": status.value,  # Use .value instead of the enum object
            "node_type": self.node_type.value,  # Use .value instead of the enum object
        }
        self.client.publish(MQTTConfig.NODE_STATUS_TOPIC, json.dumps(status_message), qos=1)


    def disconnect(self):
        """Disconnect from the MQTT broker"""
        try:
            if hasattr(self, 'client') and self.client is not None:
                self._publish_status(CLIENT_STATUS.DISCONNECTED)
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
            self.logger.info("Disconnected from MQTT broker")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    def publish_message(self, msg:dict, topic: str,qos: int = 1) -> bool:
        try:
            if not hasattr(self, 'client') or self.client is None:
                self.logger.error("Cannot publish message: MQTT client not initialized")
                return False
                
            result = self.client.publish(topic, json.dumps(msg), qos=qos)
            
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                self.logger.error(f"Failed to publish to {topic}. Error code: {result.rc}")
                return False
            self.logger.info(f"Published to {topic}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error publishing message: {str(e)}")
            return False

    def run(self):
        """Run the node's main execution loop"""
        try:          
            # Execute the concrete implementation's main loop
            self.main_loop()
            return True
        except Exception as e:
            self.logger.error(f"Error in node execution: {str(e)}")
            return False
        finally:
            self.disconnect()


    def main_loop(self):
        """Main execution loop for node"""
        while True:
            time.sleep(1)

