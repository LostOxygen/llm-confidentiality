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
import torch
from huggingface_hub import login

from framework.strategy import Strategy
from framework.attacks import (
        ATTACK_LIST, payload_splitting, obfuscation,
        indirect, manipulation, llm_attack, translation, chatml_abuse,
        masking, typoglycemia, advs_suffix
    )
from framework.defenses import (
        DEFENSES_LIST, seq_enclosure, xml_tagging, heuristic_defense,
        sandwiching, llm_eval, identity_prompt
    )
from framework.prompts import SYSTEM_PROMPTS
from framework.colors import TColors

if not os.path.isdir("/data/"):
    os.mkdir("/data/")
os.environ["TRANSFORMERS_CACHE"] = "/data/"


def main(attacks: List[str], defense: str, llm_type: str,
         temperature: float, max_level: int, create_dataset: bool
         ) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack: List[str] - specifies a list of attacks against the LLM
        defense: str - specifies the defense type
        llm_type: str - specifies the opponent LLM type
        temperature: float - specifies the opponent LLM temperature to control randomness
        max_level: int - max. system prompt level upon which to test the attacks to
        create_dataset: bool - specifies whether to create a dataset or not

    Returns:
        None
    """
    # paste the key into the key.txt file and put into the root directory
    try:
        with open(file="key.txt", mode="r", encoding="utf-8") as f:
            key = f.read().replace("\n", "")
            assert key != "", f"{TColors.FAIL}Key is empty.{TColors.ENDC}"

            os.environ["OPENAI_API_KEY"] = key
            openai.api_key = key
            print(f"{TColors.OKGREEN}OpenAI API key loaded.{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your OpenAI API key into the key.txt "
              f"file and put it into the root directory.{TColors.ENDC}")
        if llm_type in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4"]:
            sys.exit(1)

    # paste the key into the key.txt file and put into the root directory
    try:
        with open(file="hf_token.txt", mode="r", encoding="utf-8") as f:
            key = f.read().replace("\n", "")
            assert key != "", f"{TColors.FAIL}HF Token is empty.{TColors.ENDC}"

            os.environ["HF_TOKEN"] = key
            print(f"{TColors.OKGREEN}Huggingface token loaded.")
            login(token=key, add_to_git_credential=False)
            print(f"{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your Huggingface token into the hf_token.txt "
              f"file and put it into the root directory.{TColors.ENDC}")
        if llm_type in ["llama2", "llama2-7b", "llama2-13b", "llama2-70b"]:
            sys.exit(1)

    # setting devies and variables correctly
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"

    if "all" in attacks:
        attacks = ATTACK_LIST

    if max_level > len(SYSTEM_PROMPTS):
        max_level = len(SYSTEM_PROMPTS)

    print("\n"+"#"*os.get_terminal_size().columns)
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Attack Type{TColors.ENDC}: {attacks}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Defense Type{TColors.ENDC}: {defense}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Opponent LLM{TColors.ENDC}: {llm_type}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Attack Iterations{TColors.ENDC}: {max_level}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Temperature{TColors.ENDC}: {temperature}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Creating Dataset{TColors.ENDC}: {create_dataset}")
    print("#"*os.get_terminal_size().columns+"\n")

    total_successes: dict[int] = {f"{attack}" : 0 for attack in attacks}

    # initialize the strategy
    strategy = Strategy(attack_func=None, defense_func=None,
                        llm_type=llm_type, temperature=temperature,
                        max_level=max_level, create_dataset=create_dataset)

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
            case "advs_suffix": attack_func = advs_suffix
            case _:
                print(f"{TColors.FAIL}Attack type {attack} is not supported.{TColors.ENDC}")
                print(f"{TColors.FAIL}Choose from: {ATTACK_LIST}{TColors.ENDC}")
                sys.exit(1)

        # set the attack and defense functions
        strategy.set_attack_func(attack_func)
        strategy.set_defense_func(defense_func)
        # run the attack
        total_successes[attack] = strategy.execute()
        torch.cuda.empty_cache()

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
    parser.add_argument("--llm_type", "-llm", type=str, default="gpt-3.5-turbo-0301",
                        help="specifies the opponent LLM type")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="specifies the opponent LLM temperature")
    parser.add_argument("--max_level", "-m", type=int, default=10,
                        help="specifies the max system prompt level to test against")
    parser.add_argument("--create_dataset", "-cd", help="enables dataset creation",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
