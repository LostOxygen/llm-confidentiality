"""main hook to start the llm-confidentiality framework"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import sys
import datetime
import socket
import argparse
from typing import List

from src.attacks import (
        ATTACK_LIST, DEFENSES_LIST, prompt_injection, obfuscation,
        indirect, manipulation, llm_attack
    )
from src.colors import TColors


# paste the key into the key.txt file and put into the root directory
try:
    with open(file="key.txt", mode="r", encoding="utf-8") as f:
        os.environ["OPENAI_API_KEY"] = f.read()
        assert os.environ["OPENAI_API_KEY"] != "", f"{TColors.FAIL}Key is empty.{TColors.ENDC}"
        print(f"{TColors.OKGREEN}OpenAI API key loaded.{TColors.ENDC}")

except FileNotFoundError:
    print(f"{TColors.FAIL}Please paste your OpenAI API key into the key.txt " \
          f"file and put into the root directoryf{TColors.ENDC}")
    sys.exit(1)


def main(attacks: List[str], defense: str) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack: List[str] - specifies a list of attacks against the LLM
        defense: str - specifies the defense type

    Returns:
        None
    """
    print("\n\n\n"+"#"*60)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {os.cpu_count()} CPU cores on {socket.gethostname()}")
    print(f"## Attack type: {attacks}")
    print(f"## Defense type: {defense}")
    print("#"*60+"\n")

    total_successes: dict[int] = {f"{attack}" : 0 for attack in attacks}

    for attack in attacks:
        opponent_llm = None

        match attack:
            case "prompt_injection":
                if prompt_injection():
                    total_successes[attack] += 1

            case "obfuscation":
                if obfuscation():
                    total_successes[attack] += 1

            case "indirect":
                if indirect():
                    total_successes[attack] += 1

            case "manipulation":
                if manipulation():
                    total_successes[attack] += 1

            case "llm":
                if llm_attack():
                    total_successes[attack] += 1

            case _:
                print(f"{TColors.FAIL}Attack type {attack} is not supported.{TColors.ENDC}")

    print(f"{TColors.OKCYAN}Total successes: {total_successes}{TColors.ENDC}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--attacks", "-a", type=str, default=["prompt_injection"],
                        help="specifies the attack types", nargs="+", choices=ATTACK_LIST)
    parser.add_argument("--defense", "-d", type=str, default="sanitization",
                        help="specifies the defense type", choices=DEFENSES_LIST)
    args = parser.parse_args()

    main(**vars(args))
