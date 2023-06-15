"""library for attack implementations and helper functions"""

from typing import List, Final

ATTACK_LIST: Final[List[str]] = ["prompt_injection", "obfuscation", "indirect",
                                 "manipulation", "llm"]

DEFENSES_LIST: Final[List[str]] = ["prompt_based", "sanitization", "advs_training",
                                   "sandboxin"]
