"""main hook to start the llm-confidentiality framework"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import sys
import datetime
import socket
import argparse
from typing import List

import openai

from src.strategy import Strategy
from src.attacks import (
        ATTACK_LIST, payload_splitting, obfuscation,
        indirect, manipulation, llm_attack, translation, chatml_abuse,
        masking, typoglycemia
    )
from src.defenses import (
        DEFENSES_LIST, seq_enclosure, xml_tagging, heuristic_defense,
        sandwiching, llm_eval, identity_prompt
    )
from src.prompts import SYSTEM_PROMPTS
from src.colors import TColors


def main(attacks: List[str], defense: str, opponent_type: str,
         temperature: float, max_level: int,
         ) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack: List[str] - specifies a list of attacks against the LLM
        defense: str - specifies the defense type
        opponent_type: str - specifies the opponent LLM type
        temperature: float - specifies the opponent LLM temperature to control randomness
        max_level: int - max. system prompt level upon which to test the attacks to

    Returns:
        None
    """
    # paste the key into the key.txt file and put into the root directory
    try:
        with open(file="key.txt", mode="r", encoding="utf-8") as f:
            key = f.read()
            assert key != "", f"{TColors.FAIL}Key is empty.{TColors.ENDC}"

            os.environ["OPENAI_API_KEY"] = key
            openai.api_key = key
            print(f"{TColors.OKGREEN}OpenAI API key loaded.{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your OpenAI API key into the key.txt "
              f"file and put it into the root directory.{TColors.ENDC}")
        sys.exit(1)


    if "all" in attacks:
        attacks = ATTACK_LIST

    if max_level > len(SYSTEM_PROMPTS):
        max_level = len(SYSTEM_PROMPTS)

    print("\n"+"#"*60)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {os.cpu_count()} CPU cores on {socket.gethostname()}")
    print(f"## Attack type: {attacks}")
    print(f"## Defense type: {defense}")
    print(f"## Opponent LLM: {opponent_type}")
    print(f"## Testing Level 0-{max_level}")
    print(f"## Temperature: {temperature}")
    print("#"*60+"\n")

    total_successes: dict[int] = {f"{attack}" : 0 for attack in attacks}

    # set the defense function
    match defense:
        case "seq_enclosure": defense_func = seq_enclosure
        case "xml_tagging": defense_func = xml_tagging
        case "heuristic_defense": defense_func = heuristic_defense
        case "sandwiching": defense_func = sandwiching
        case "llm_eval": defense_func = llm_eval
        case "None": defense_func = identity_prompt
        case _: defense_func = identity_prompt

    for attack in attacks:
        # set the attack function
        match attack:
            case "payload_splitting": attack_func = payload_splitting
            case "obfuscation": attack_func = obfuscation
            case "indirect": attack_func = indirect
            case "manipulation": attack_func = manipulation
            case "llm": attack_func = llm_attack
            case "translation": attack_func = translation
            case "chatml_abuse": attack_func = chatml_abuse
            case "masking": attack_func = masking
            case "typoglycemia": attack_func = typoglycemia
            case _:
                print(f"{TColors.FAIL}Attack type {attack} is not supported.{TColors.ENDC}")
                print(f"{TColors.FAIL}Choose from: {ATTACK_LIST}{TColors.ENDC}")
                sys.exit(1)

        # initialize the strategy
        strategy = Strategy(attack_func=attack_func, defense_func=defense_func,
                            llm_type=opponent_type, temperature=temperature,
                            max_level=max_level)

        # run the attack
        total_successes[attack] = strategy.execute()


    # print the results
    print(f"{TColors.OKBLUE}{TColors.BOLD}>> Attack Results:{TColors.ENDC}")
    for attack, successes in total_successes.items():
        print(f"Attack: {TColors.OKCYAN}{attack}{TColors.ENDC} - Successes: {successes}/"
              f"{max_level}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--attacks", "-a", type=str, default=["payload_splitting"],
                        help="specifies the attack types", nargs="+")
    parser.add_argument("--defense", "-d", type=str, default="None",
                        help="specifies the defense type", choices=DEFENSES_LIST)
    parser.add_argument("--opponent_type", "-o", type=str, default="gpt-3.5-turbo-0301",
                        help="specifies the opponent LLM type")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="specifies the opponent LLM temperature")
    parser.add_argument("--max_level", "-m", type=int, default=10,
                        help="specifies the max system prompt level to test against")
    args = parser.parse_args()

    main(**vars(args))
