"""library for defense implementations and helper functions"""
from typing import Final, List


DEFENSES_LIST: Final[List[str]] = ["seq_enclosure", "xml_tagging", "heuristic_defense",
                                   "sandwiching", "llm_eval", "identity_prompt"]


def identity_prompt(prompt: str) -> None:
    """
    Uses the prompt as is without any modifications.
    """
    return prompt


def seq_enclosure(prompt: str) -> None:
    """
    Encloses the prompt in a sequence of random characters.
    
    Parameters:
        prompt (str): the prompt to be enclosed

    Returns:
        defense_prompt: str - the prompt enclosed in a sequence of random characters
    """
    defense_prompt = ""

    return defense_prompt


def xml_tagging(prompt: str) -> None:
    """
    Encloses the user input in prompts in XML tags to help the LLM evade adversarial inputs.
    
    Parameters:
        prompt (str): the prompt to be enclosed

    Returns:
        defense_prompt: str - the prompt enclosed in XML tags
    """
    defense_prompt = ""

    return defense_prompt


def heuristic_defense(prompt: str) -> None:
    """
    Uses rule based heuristics to detect and remove adversarial prompt inputs.
    
    Parameters:
        prompt (str): the prompt to be enclosed

    Returns:
        defense_prompt: str - the sanitized prompt (can be empty)
    """
    defense_prompt = ""

    return defense_prompt


def sandwiching(prompt: str) -> None:
    """
    Encloses the user input prompt between two other prompts to help the LLM evade advs. inputs.
    
    Parameters:
        prompt (str): the prompt to be enclosed

    Returns:
        defense_prompt: str - the prompt enclosed in two other prompts
    """
    defense_prompt = ""

    return defense_prompt


def llm_eval(prompt: str) -> None:
    """
    Uses a separate LLM to evaluate the prompt and remove adversarial inputs.
    
    Parameters:
        prompt (str): the prompt to be enclosed

    Returns:
        defense_prompt: str - the evaluated and sanitized prompt
    """
    defense_prompt = ""

    return defense_prompt
