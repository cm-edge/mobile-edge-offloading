from shared.server_node import ServerNode
from shared.config import MQTTConfig, NODE_TYPE

def main():
    edge = ServerNode(
        node_type=NODE_TYPE.EDGE,
        broker_host=MQTTConfig.BROKER_HOST_EDGE,
        broker_port=MQTTConfig.BROKER_PORT,
    )
    edge.run()


if __name__ == "__main__":
    main()
