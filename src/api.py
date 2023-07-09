"""API class implementation to interact with the ChatVisuzalization class"""
from datetime import datetime
from typing import Final
import os

PATH: Final[str] = "/tmp/llm_chat/"

class ChatAPI:
    """socket API class to interact with ChatVisualization class"""
    def __init__(self):
        pass

    @staticmethod
    def add_message(user: str, message: str):
        """adds messages to the chat visualization"""
        # check if the paths are valid
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        timestamp = datetime.now().strftime("%d. %B %Y %I:%M%p")
        final_msg = f"MSG|{user}|{message}|{timestamp}@"
        final_msg.encode("utf-8")

        with open(PATH+"chat_log.txt", "a", encoding="utf-8") as chat_file:
            chat_file.write(final_msg)
