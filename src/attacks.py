"""library for attack implementations and helper functions"""

from typing import List, Final

ATTACK_LIST: Final[List[str]] = ["prompt_injection", "obfuscation", "indirect",
                                 "manipulation", "llm"]

DEFENSES_LIST: Final[List[str]] = ["prompt_based", "sanitization", "advs_training",
                                   "sandboxing"]


def prompt_injection() -> bool:
    """
    implementation of a prompt injection attack

    Parameters:
        None
    
    Returns:
        success: bool - whether the attack was successful or not
    """
    pass


def obfuscation() -> bool:
    """
    implementation of an obfuscation attack

    Parameters:
        None
    
    Returns:
        success: bool - whether the attack was successful or not
    """
    pass


def indirect() -> bool:
    """
    implementation of an indirect attack

    Parameters:
        None
    
    Returns:
        success: bool - whether the attack was successful or not
    """
    pass


def manipulation() -> bool:
    """
    implementation of a manipulation attack

    Parameters:
        None
    
    Returns:
        success: bool - whether the attack was successful or not
    """
    pass


def llm_attack() -> bool:
    """
    implementation of an llm attack

    Parameters:
        None
    
    Returns:
        success: bool - whether the attack was successful or not
    """
    pass
