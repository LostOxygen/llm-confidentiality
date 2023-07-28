"""API class implementation to interact with the ChatVisuzalization class"""
from datetime import datetime
from typing import Final, List
import os

PATH: Final[str] = "/tmp/llm_chat/"

class ChatAPI:
    """API class to interact with the ChatVisualization class via temp files"""
    def __init__(self):
        pass

    @staticmethod
    def add_message(user: str, message: str) -> None:
        """adds messages to the chat visualization"""
        # check if the paths are valid
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        timestamp = datetime.now().strftime("%d. %B %Y %I:%M%p")
        final_msg = f"MSG|{user}|{message}|{timestamp}@"
        final_msg.encode("utf-8")

        with open(PATH+"chat_log.txt", "a", encoding="utf-8") as chat_file:
            chat_file.write(final_msg)


    @staticmethod
    def add_input_message(message: str) -> None:
        """adds messages to the chat visualization"""
        # check if the paths are valid
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        message += "\n"
        message.encode("utf-8")

        with open(PATH+"chat_input_log.txt", "a", encoding="utf-8") as chat_file:
            chat_file.write(message)


    @staticmethod
    def get_first_message(file_name: str = "chat_input_log.txt") -> str:
        """returns the first (oldest) message in the input chat log and removes it from the log"""
        with open(PATH+file_name, "r", encoding="utf-8") as chat_file:
            complete_log = chat_file.readlines()
            first_line = complete_log.pop(0) # remove the first line from the log

        with open(PATH+file_name, "w", encoding="utf-8") as chat_file:
            chat_file.write(complete_log)

        return first_line


    @staticmethod
    def get_all_messages(file_name: str = "chat_input_log.txt") -> List[str]:
        """returns all messages in the input chat log and clears the log"""
        with open(PATH+file_name, "r", encoding="utf-8") as chat_file:
            complete_log = chat_file.readlines()

        return complete_log
