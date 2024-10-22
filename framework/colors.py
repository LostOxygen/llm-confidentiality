"""helper library for printing colored text"""
from typing import Final

class TColors():
    """Color class with ANSI escape codes for colored text"""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


ATTACK_NAMES: Final[dict[str, str]] = {
    "chat_base": "Chat Baseline Questions",
    "payload_splitting": "Payload Splitting",
    "obfuscation": "Base16/32/64/85 Obfuscation",
    "indirect": "Indirect",
    "jailbreak": "Jailbreak",
    "llm": "LLM",
    "translation": "Translation",
    "chatml_abuse": "ChatML-Abuse",
    "masking": "Masking",
    "typoglycemia": "Typoglycemia",
    "advs_suffix": "Adversarial Suffix",
    "identity": "Identity",
    "base_attack": "Base Attack",
    "prefix_injection": "Prefix Injection",
    "refusal_suppression": "Refusal Suppression",
    "context_ignoring": "Context Ignoring",
    "context_termination": "Context Termination",
    "context_switching_separators": "Context Switching Separators",
    "few_shot": "Few-Shot",
    "cognitive_hacking": "Cognitive Hacking"
}

DEFENSE_NAMES: Final[dict[str, str]] = {
    "seq_enclosure": "Sequence Enclosure",
    "xml_tagging": "XML Tagging",
    "heuristic_defense": "Heuristic Defense",
    "llm_eval": "LLM Evaluation",
    "ppl_detection": "PPL Detection",
    "prompt_guard": "PromptGuard",
    "sandwiching": "Sandwiching",
    "identity_prompt": "Identity"
}
