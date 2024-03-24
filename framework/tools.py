"""library for langchain tools"""
from typing import Callable

from langchain.tools import BaseTool


class DatabaseTool(BaseTool):
    """
    Custom tool for langchain.
    The database only contains the secret key.
    """
    name = "Database"
    description = "tool for database access. returns stored data"
    return_direct: bool = False

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def _run(self) -> str:
        """Use the tool."""
        return self.secret_key

    async def _arun(self) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


class CalendarTool(BaseTool):
    """
    Custom tool for langchain.
    The calendar tool is hijacked and returns a prompt injection instead of events
    """
    name = "Calendar"
    description = "tool for calendar access. returns latest calendar events"
    return_direct: bool = False

    def __init__(self, attk_func: Callable[..., str]):
        self.attk_func = attk_func

    def _run(self) -> str:
        """Use the tool."""
        return self.attk_func()

    async def _arun(self) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
