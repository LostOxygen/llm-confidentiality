"""main hook to start the llm-confidentiality framework"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import sys
import datetime
import socket
import argparse
from typing import List

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from src.attacks import (
        ATTACK_LIST, DEFENSES_LIST, payload_splitting, obfuscation,
        indirect, manipulation, llm_attack
    )
from src.prompts import SYSTEM_PROMPTS
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


def main(attacks: List[str], defense: str, opponent_type: str) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack: List[str] - specifies a list of attacks against the LLM
        defense: str - specifies the defense type
        opponent_type: str - specifies the opponent LLM type

    Returns:
        None
    """
    print("\n"+"#"*60)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {os.cpu_count()} CPU cores on {socket.gethostname()}")
    print(f"## Attack type: {attacks}")
    print(f"## Defense type: {defense}")
    print(f"## Opponent LLM: {opponent_type}")
    print("#"*60+"\n")

    total_successes: dict[int] = {f"{attack}" : 0 for attack in attacks}

    for attack in attacks:
        opponent_llm = ChatOpenAI(temperature=0.7, model_name=opponent_type)

        match attack:
            case "payload_splitting":
                attack_successes = payload_splitting(opponent_llm)
                total_successes[attack] += attack_successes

            case "obfuscation":
                attack_successes = obfuscation(opponent_llm)
                total_successes[attack] += attack_successes

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

        del opponent_llm

    # print the results
    print(f"{TColors.OKBLUE}{TColors.BOLD}>> Attack Results:{TColors.ENDC}")
    for attack, successes in total_successes.items():
        print(f"{TColors.OKCYAN}Attack: {attack} - Successes: {successes}/" \
              f"{len(SYSTEM_PROMPTS)}{TColors.ENDC}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--attacks", "-a", type=str, default=["payload_splitting"],
                        help="specifies the attack types", nargs="+", choices=ATTACK_LIST)
    parser.add_argument("--defense", "-d", type=str, default="sanitization",
                        help="specifies the defense type", choices=DEFENSES_LIST)
    parser.add_argument("--opponent_type", "-o", type=str, default="gpt-3.5-turbo",
                        help="specifies the opponent LLM type")
    args = parser.parse_args()

    main(**vars(args))
