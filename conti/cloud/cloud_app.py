from shared.server_node import ServerNode
from shared.config import MQTTConfig, NODE_TYPE
def main():
    cloud = ServerNode(
        node_type=NODE_TYPE.CLOUD,
        broker_host=MQTTConfig.BROKER_HOST_CLOUD,
        broker_port=MQTTConfig.BROKER_PORT,
    )
    cloud.run()


if __name__ == "__main__":
    main()
