"""library for LLM models, functions and helper stuff"""

from abc import ABC, abstractmethod

class LLM(ABC):
    """abstract implementation of a genereric LLM model"""

    @abstractmethod
    def predict(self, text: str) -> str:
        """
        predicts a response for a given text input

        Parameters:
            text: str - the text to predict a response for

        Returns:
            response: str - the response
        """
        pass
