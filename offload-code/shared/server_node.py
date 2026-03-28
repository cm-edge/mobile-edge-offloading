import json
import time
from shared.node import Node
from shared.message_processor import process_message
from shared.config import MQTTConfig, NODE_TYPE

class ServerNode(Node):
    """
    ServerNode class that extends the Node base class.
    This class serves as a common base for both Edge and Cloud servers,
    implementing shared message processing functionality.
    """
    
    def __init__(self, node_type: NODE_TYPE, broker_host, broker_port):
        """Initialize the server node with specific configuration"""
        super().__init__(
            node_type=node_type,
            broker_host=broker_host,
            broker_port=broker_port,
        )
        
    def on_message(self, client, userdata, msg):
        """Process incoming messages and send responses back to devices"""
        try:
            # Parse incoming message
            loaded_msg = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            self.logger.error("Received malformed JSON message")
            return None  # Stop processing this message

        try:
            device_id = loaded_msg.get('client_id', 'unknown')
            request_id = loaded_msg.get('request_id', 'unknown')
            payload_type = loaded_msg.get('payload_type', 'unknown')
            
            # Log message receipt (specific format handled by subclasses)
            self._log_message_receipt(device_id, request_id, payload_type)

            # Process the message using the shared processor
            result_msg = process_message(loaded_msg)

            # Construct client-specific response topic using config
            response_topic = MQTTConfig.get_response_topic(device_id)

            self.publish_message(msg=result_msg, topic=response_topic)
            self.logger.info(
                f"Replying request[{request_id}] -> topic[{response_topic}]")
                
        except Exception as e:
            self.logger.error(f"Error processing request[{request_id}]: {e}")
    
    def _log_message_receipt(self, device_id, request_id, payload_type):
        """Customized log format for server node"""
        self.logger.info(
            f"Received request[{request_id}]-type[{payload_type}] <- {device_id}")
