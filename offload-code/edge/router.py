import json
import random
from datetime import datetime

from shared.node import Node
from shared.config import MQTTConfig, RouterConfig, NODE_TYPE


class Router(Node):
    """
    Router class that extends the Node class.
    This class is responsible for handling message routing between devices, edge, and cloud.
    """
    
    def __init__(self, 
                node_type=NODE_TYPE.ROUTER, 
                broker_host=MQTTConfig.BROKER_HOST_ROUTER,
                broker_port=MQTTConfig.BROKER_PORT,
                 ):
        super().__init__(
            node_type=node_type,
            broker_host=broker_host,
            broker_port=broker_port,
        )
        # Routing configuration
        self.edge_probability = RouterConfig.EDGE_PROBABILITY
        self.edge_requests_topic = MQTTConfig.EDGE_REQUESTS_TOPIC
        self.cloud_requests_topic = MQTTConfig.CLOUD_REQUESTS_TOPIC
        
    def route_message(self, message) -> tuple[dict, str]:
        """
        Determine whether the message should be processed locally by the edge
        or forwarded to the cloud based on routing logic.
        
        Args:
            message: The message payload to route
            
        Returns:
            tuple: (enriched message with routing metadata, routing decision)
        """
        # Mock routing algorithm using random
        should_process_on_edge = random.random() < self.edge_probability
        
        # In a real implementation, you would examine the message content
        # to make an intelligent routing decision based on:
        # - Message priority/urgency
        # - Current edge load
        # - Available edge capabilities
        # - Required processing power
        # - Network conditions
        # - Data sensitivity
        
        routing_decision = "edge" if should_process_on_edge else "cloud"
        
        # Add routing metadata
        message["routing"] = {
            "decision": routing_decision,
            "router_id": self.client_id,
            "routed_at": datetime.now().isoformat()
        }
        
        return message, routing_decision

    def on_message(self, client, userdata, msg):
        """Process incoming messages and route them appropriately"""
        try:
            # Parse incoming message 
            payload = json.loads(msg.payload.decode())
            request_id = payload.get('request_id', 'unknown')
            device_id = payload.get('client_id', 'unknown')
            payload_type = payload.get('payload_type', 'unknown')
            self.logger.info(f"Received request[{payload_type}-{request_id}] <- {device_id}")

            # Determine routing
            routed_message, decision = self.route_message(payload)

            # Route the message accordingly
            destination = self.edge_requests_topic if decision == "edge" else self.cloud_requests_topic
            self.publish_message(msg=routed_message, topic=destination)
            self.logger.info(f"Routed request[{request_id}] -> {destination}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
