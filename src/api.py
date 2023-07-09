"""API class implementation to interact with the ChatVisuzalization class"""
from datetime import datetime
from typing import Final
import os

PATH: Final[str] = "/tmp/llm_chat/"

class ChatAPI:
    """socket API class to interact with ChatVisualization class"""
    def __init__(self):
        # check if the paths are valid
        if not os.path.exists(PATH):
            os.makedirs(PATH)


    def add_message(self, user: str, message: str):
        """adds messages to the chat visualization"""
        timestamp = datetime.now().strftime("%d. %B %Y %I:%M%p")
        final_msg = f"MSG|{user}|{message}|{timestamp}@"
        final_msg.encode("utf-8")

        with open(PATH+"chat_log.txt", "a", encoding="utf-8") as chat_file:
            chat_file.write(final_msg)
