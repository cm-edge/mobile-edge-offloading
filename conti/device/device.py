from shared.node import Node
from shared.config import MQTTConfig, AppConfig, PathConfig, NODE_TYPE, PAYLOAD_TYPE, ImageConfig
import json
import random
import base64
import uuid
from pathlib import Path
import time
import os  

class Device(Node):
    """
    Device class that extends the Node class.
    Handles device-specific functionality.
    """
    def __init__(
        self,
        node_type: NODE_TYPE = NODE_TYPE.DEVICE,
        broker_host: str | None = None,
    ):
        # Read the running mode (default offload)
        self.run_mode = os.getenv("RUN_MODE", "offload").lower()

        # Allow overriding broker settings with environment variables (priority: parameters > environment variables > configuration)
        env_host = os.getenv("MQTT_BROKER_HOST", MQTTConfig.BROKER_HOST_DEVICE)
        broker_host = broker_host or env_host

        # Initialize first (Node will create logger, client, etc.)
        super().__init__(
            node_type=node_type,
            broker_host=broker_host,
            broker_port=MQTTConfig.BROKER_PORT,
        )

        self.publish_topic = MQTTConfig.DEVICE_REQUESTS_TOPIC

        # Skip MQTT when local: If the parent class has already attempted to connect, disconnect immediately and set it to null
        if self.run_mode == "local":
            try:
                if getattr(self, "client", None):
                    try:
                        self.client.loop_stop()
                    except Exception:
                        pass
                    try:
                        self.client.disconnect()
                    except Exception:
                        pass
            finally:
                self.client = None
                self.logger.info("RUN_MODE=local → Skip MQTT initialization/connection")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.logger.info(
                f"Received response[{payload.get('request_id')}] <- processor[{payload.get('processed_by')}]"
            )
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def get_random_image(self) -> str | None:
        """
        Load a random image from the data folder and return it as a base64 string.
        """
        data_folder = Path(PathConfig.DATA_FOLDER)  # Usually device/data/
        try:
            if not data_folder.exists():
                self.logger.error(f"Data folder not found: {data_folder}")
                return None

            image_files = [
                f for f in data_folder.glob("*")
                if f.is_file() and f.suffix.lower() in ImageConfig.VALID_IMAGE_EXTENSIONS
            ]
            if not image_files:
                self.logger.error(f"No image files found in {data_folder}")
                return None

            image_path = random.choice(image_files)
            self.logger.debug(f"Selected image: {image_path.name}")

            with open(image_path, "rb") as img_file:
                encoded_str = base64.b64encode(img_file.read()).decode("utf-8")
                self.logger.info(f"Encoded image length: {len(encoded_str)} chars")
            return encoded_str
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None

    def main_loop(self):
        run_mode = os.getenv("RUN_MODE", "offload")  # "local" or "offload"
        max_iters = int(os.getenv("MAX_ITERS", "0"))  # 0 = infinite loop
        counter = 0

        while True:
            encoded_image_str = self.get_random_image()
            if not encoded_image_str:
                self.logger.warning("No image found; retrying...")
                time.sleep(AppConfig.MSG_INTERVAL)
                continue

            req_id = uuid.uuid4().hex

            if run_mode == "local":
                from shared.message_processor import process_request
                payload = {
                    "client_id": self.client_id,
                    "request_id": req_id,
                    "payload_type": PAYLOAD_TYPE.IMAGE.value,
                    "image_b64": encoded_image_str,
                }
                result = process_request(payload)
                self.logger.info(
                    f"[LOCAL] req={req_id} top1={result.get('top1')} timing={result.get('timing_ms')}ms"
                )
            else:
                msg = {
                    "client_id": self.client_id,
                    "request_id": req_id,
                    "payload_type": PAYLOAD_TYPE.IMAGE.value,
                    "image_b64": encoded_image_str,
                }
                self.publish_message(msg=msg, topic=self.publish_topic)
                self.logger.info(f"[OFFLOAD] published req={req_id} to {self.publish_topic}")

            counter += 1
            if max_iters > 0 and counter >= max_iters:
                self.logger.info(f"Reached MAX_ITERS={max_iters}, exiting loop.")
                break

            time.sleep(AppConfig.MSG_INTERVAL)
