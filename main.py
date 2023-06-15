"""main hook to start the llm-confidentiality framework"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import datetime
import socket
import argparse

from src.attacks import ATTACK_LIST, DEFENSES_LIST

def main(attack_type: str, defense_type: str) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack_type: str - specifies the attack type

    Returns:
        None
    """
    print("\n\n\n"+"#"*60)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {os.cpu_count()} CPU cores on {socket.gethostname()}")
    print(f"## Attack type: {attack_type}")
    print(f"## Defense type: {defense_type}")
    print("#"*60+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--attack_type", type=str, default="prompt_injection",
                        help="specifies the attack type", choices=ATTACK_LIST)
    parser.add_argument("--defense_type", type=str, default="sanitization",
                        help="specifies the defense type", choices=DEFENSES_LIST)
    args = parser.parse_args()

    main(**vars(args))
