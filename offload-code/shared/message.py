import uuid
import json
# message type enum
from enum import Enum

class MessageType(Enum):
    """
    Enum representing the type of message.
    """
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"


class Message:
    """
    An abstract base class for messages in the system.
    This class defines the basic structure and methods that all message types
    Image, Text based messages should implement and inherit from.
    """
    def __init__(self, client_id: str, payload_type: MessageType):
        self.client_id = client_id
        self.request_id = uuid.uuid4().hex
        self.payload_type = payload_type
    

class ImageMessage(Message):
    """
    A class representing an image message.
    This class extends the Message class and adds functionality specific to image messages.
    """

    def __init__(self, client_id: str):
        super().__init__(client_id, payload_type=MessageType.IMAGE)

    def payload(self, image_data: str) -> 'ImageMessage':
        """
        Set the payload for the image message.
        """
        self.image_data = image_data
        return self

    def serialize(self) -> str:
        """
        Serialize the image message to a JSON-compatible format.
        """
        return json.dumps({
            "client_id": self.client_id,
            "request_id": self.request_id,
            "payload_type": self.payload_type.value,
            "image_data": self.image_data
        })
