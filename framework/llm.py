"""library for LLM models, functions and helper stuff"""
from openai import ChatCompletion

class LLM():
    """abstract implementation of a genereric LLM model"""
    def __init__(self, llm_type: str, temperature: float = 1.0):
        self.llm_type: str = llm_type
        self.temperature: float = temperature


    def predict(self, system_prompt: str, user_prompt: str) -> str:
        """
        predicts a response for a given prompt input

        Parameters:
            system_prompt: str - the system prompt to initialize the LLM
            user_prompt: str - the user prompt for the LLM to respond on

        Returns:
            response: str - the LLMs' response
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        match self.llm_type:
            case ("gpt-3.5-turbo" | "gpt-3.5-turbo-0301" | "gpt-4"):
                completion = ChatCompletion.create(model=self.llm_type,
                                                   messages=messages,
                                                   temperature=self.temperature)
                response = completion.choices[0].message.content
            case _:
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")

        return response
