"""main hook to start the llm-confidentiality framework"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import sys
import time
import datetime
import socket
import argparse
from typing import List

import openai
import torch
from huggingface_hub import login

from framework.strategy import AttackStrategy
from framework.attacks import (
        ATTACK_LIST,
        payload_splitting,
        obfuscation,
        jailbreak,
        translation,
        chatml_abuse,
        masking,
        typoglycemia,
        advs_suffix,
        chat_base
    )
from framework.defenses import (
        DEFENSES_LIST,
        seq_enclosure,
        xml_tagging,
        heuristic_defense,
        sandwiching,
        llm_eval,
        identity_prompt,
        ppl_detection
    )
from framework.colors import TColors
from framework.utils import log_results


if not os.path.isdir("/data/"):
    os.mkdir("/data/")
os.environ["TRANSFORMERS_CACHE"] = "/data/"


def main(
        attacks: List[str],
        defenses: List[str],
        llm_type: str,
        temperature: float,
        iterations: int,
        create_prompt_dataset: bool,
        create_response_dataset: bool,
        name_suffix: str
    ) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack: List[str] - specifies a list of attacks against the LLM
        defenses: List[str] - specifies the defense type
        llm_type: str - specifies the opponent LLM type
        temperature: float - specifies the opponent LLM temperature to control randomness
        iterations: int - number of attack iterations to test system prompts against
        create_prompt_dataset: bool - specifies whether to create a system prompt dataset or not
        create_response_dataset: bool - specifies whether to create a responses dataset or not
        name_suffix: str - adds a name suffix for loading custom models

    Returns:
        None
    """
    start = time.perf_counter()  # start timer

    # paste the OpenAI key into the key.txt file and put into the root directory
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

    # paste the Huggingface token into the hf_token.txt file and put into the root directory
    try:
        with open(file="hf_token.txt", mode="r", encoding="utf-8") as f:
            key = f.read().replace("\n", "")
            assert key != "", f"{TColors.FAIL}HF Token is empty.{TColors.ENDC}"

            os.environ["HF_TOKEN"] = key
            print(f"{TColors.OKGREEN}Huggingface token loaded.")
            login(token=key, add_to_git_credential=True)
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
    if "all" in defenses:
        defenses = ["none"] + DEFENSES_LIST

    # add '-' in front of the name suffix
    if name_suffix != "" and not name_suffix.startswith("-"):
        name_suffix = "-" + name_suffix

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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Defense Type{TColors.ENDC}: {defenses}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Opponent LLM{TColors.ENDC}: " \
          f"{TColors.HEADER}{llm_type}{TColors.OKCYAN}{name_suffix}{TColors.ENDC}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Attack Iterations{TColors.ENDC}: {iterations}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Temperature{TColors.ENDC}: {temperature}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Creating System Prompt Dataset{TColors.ENDC}: " \
          f"{create_prompt_dataset}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Creating Responses Dataset{TColors.ENDC}: " \
          f"{create_response_dataset}")
    print("#"*os.get_terminal_size().columns+"\n")

    total_successes: dict[int] = {f"{attack}" : 0 for attack in attacks}

    # initialize the strategy
    strategy = AttackStrategy(
                attack_func=None,
                defense_func=None,
                llm_type=llm_type,
                llm_suffix=name_suffix,
                temperature=temperature,
                iterations=iterations,
                create_prompt_dataset=create_prompt_dataset,
                create_response_dataset=create_response_dataset
            )

    for defense in defenses:
        # set the defense function
        match defense:
            case "seq_enclosure": defense_func = seq_enclosure
            case "xml_tagging": defense_func = xml_tagging
            case "heuristic_defense": defense_func = heuristic_defense
            case "sandwiching": defense_func = sandwiching
            case "llm_eval": defense_func = llm_eval
            case "ppl_detection": defense_func = ppl_detection
            case ("None" | "none"): defense_func = identity_prompt
            case _: defense_func = identity_prompt

        for attack in attacks:
            # set the attack function
            match attack:
                case "chat_base": attack_func = chat_base
                case "payload_splitting": attack_func = payload_splitting
                case "obfuscation": attack_func = obfuscation
                case "jailbreak": attack_func = jailbreak
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

        # print and log the results
        print(f"{TColors.OKBLUE}{TColors.BOLD}>> Attack Results:{TColors.ENDC}")
        for attack, successes in total_successes.items():
            print(f"Attack: {TColors.OKCYAN}{attack}{TColors.ENDC} - Successes: {successes}/"
                f"{iterations}")
        log_results(
                llm_name=llm_type+name_suffix,
                defense_name=defense,
                success_dict=total_successes,
                iters=iterations
            )

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"{TColors.HEADER}Computation Time: {duration}{TColors.ENDC}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--attacks", "-a", type=str, default=["payload_splitting"],
                        help="specifies the attack types", nargs="+")
    parser.add_argument("--defenses", "-d", type=str, default=["none"],
                        help="specifies the defense type", nargs="+")
    parser.add_argument("--llm_type", "-llm", type=str, default="gpt-3.5-turbo-0301",
                        help="specifies the opponent LLM type")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="specifies the opponent LLM temperature")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                        help="specifies the number of iterations to test systems prompts")
    parser.add_argument("--create_prompt_dataset", "-cp", help="enabl. sys prompt dataset creation",
                        action="store_true", default=False)
    parser.add_argument("--create_response_dataset", "-cr", help="enabl. response dataset creation",
                        action="store_true", default=False)
    parser.add_argument("--name_suffix", "-n", help="adds a name suffix for loading custom models",
                        default="", type=str)
    args = parser.parse_args()
    main(**vars(args))
