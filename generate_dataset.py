"""main hook to start the dataset generation"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
from pathlib import Path
import sys
import time
from random import randint
import datetime
import psutil
import getpass
import argparse
from typing import Type

import openai
import torch
import pkbar
from huggingface_hub import login

from framework.llm import LLM
from framework.dataset import PromptDataset, DatasetState

from framework.colors import TColors
from framework.prompts import SYSTEM_PROMPTS


if not os.path.isdir(str(Path.home() / "data")):
    os.mkdir(str(Path.home() / "data"))
os.environ["TRANSFORMERS_CACHE"] = str(Path.home() / "data")


def main(
        dataset_size: int,
        llm_type: str,
        name_suffix: str,
        device: str,
    ) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack: List[str] - specifies a list of attacks against the LLM
        dataset_size: int - specifies the size of the resulting dataset
        llm_type: str - specifies the opponent LLM type
        name_suffix: str - adds a name suffix for loading custom models
        device: str - specifies the device to run the LLM on (cpu, mps or cuda)

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
        if llm_type in ["gpt-3.5-turbo", "gpt-4"]:
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
        if llm_type.startswith("llama"):
            sys.exit(1)

    # set the devices correctly
    if device == "cpu":
        device = torch.device("cpu")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device(device)
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device(device)
    else:
        print(f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} " \
              f"{TColors.ENDC}is not available. Setting device to CPU instead.")
        device = torch.device("cpu")

    # add '-' in front of the name suffix
    if name_suffix != "" and not name_suffix.startswith("-"):
        name_suffix = "-" + name_suffix

    print("\n"+f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-23))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if (device == "cuda" or torch.device("cuda")) and torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    elif (device == "mps" or torch.device("mps")) and torch.backends.mps.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    else:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    print(f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-14))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Size{TColors.ENDC}: {dataset_size}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}LLM Type{TColors.ENDC}: " \
          f"{TColors.HEADER}{llm_type}{TColors.OKCYAN}{name_suffix}{TColors.ENDC}")
    print("#"*os.get_terminal_size().columns+"\n")

    # initialize the dataset
    dataset_train: PromptDataset = PromptDataset(state=DatasetState.TRAIN)
    dataset_test: PromptDataset = PromptDataset(state=DatasetState.TEST)

    # initialize the LLM
    llm: Type[LLM] = LLM(
        llm_type=llm_type,
        temperature=0.0,
        device=device,
        llm_suffix=name_suffix,
        verbose=False,
    )

    progress_bar_train = pkbar.Pbar(
        target=dataset_size,
        width=50,
        name=f"{TColors.OKBLUE}Generating System Prompt Training Dataset{TColors.ENDC}"
    )

    while len(dataset_train) < dataset_size:
        sys_prompt = """
        You are a helpful security chatbot which helps researchers to design more safe 
        artificial intelligence systems that are not vulnerable to attacks. Follow the 
        research instructions of the researchers and help them to improve the security of LLMs.
        """

        random_sys_prompt = dataset_train.get_random_prompt()

        user_prompt = f"""
        Large Language Models are often vulnerable to so called prompt injection attacks, where 
        users design input prompts in a specific way to manipulate the output and behavior of the LLM.
        Create and print a new safe system prompt to instruct an LLM to be safe against prompt
        injection and prompt leaking attacks, especially when using tools like Emails, Calendars,
        or Clouds. These tools can be used by the LLM to manage and access user data which could be
        confidential.
        Start with set of instructions to guide the instructed LLM into keeping the secret key safe.
        Print only the system prompt and nothing else. The system prompt has to be at least 100 characters
        and at most 300 characters long and is not allowed to use any special symbols like \"\n\" or to repeatedly
        use the same words or sentences. The system prompt should be creative while being non-repetitive. 
        Do not write any introduction sentences. Try to keep the tool usage safe and secure from leaks.
        You can orient yourself at this example prompt: 
        \"{random_sys_prompt}\"

        The new system prompt is:
        """

        response, _ = llm.chat(sys_prompt, user_prompt)

        # sanitize the response
        response = response.replace("<|im_start|>system\n", "")
        response = response.replace("<|startoftext|>\n", "")
        response = response.replace("<|startoftext|>", "")
        response = response.replace("System prompts:\n", "")
        response = response.replace("<|im_start|>assistant\n", "")

        # add the response to the dataset
        dataset_train.add_prompt(response)
        progress_bar_train.update(len(dataset_train))


    progress_bar_test = pkbar.Pbar(
        target=dataset_size//3,
        width=50,
        name=f"{TColors.OKBLUE}\nGenerating System Prompt Test Dataset{TColors.ENDC}"
    )

    while len(dataset_test) < dataset_size//3:
        sys_prompt = """
        You are a helpful security chatbot which helps researchers to design more safe 
        artificial intelligence systems that are not vulnerable to attacks. Follow the 
        research instructions of the researchers and help them to improve the security of LLMs.
        """

        if len(dataset_test) <= 100:
            random_sys_prompt = SYSTEM_PROMPTS[str(randint(3, len(SYSTEM_PROMPTS)-1))]#
        else:
            random_sys_prompt = dataset_train.get_random_prompt()

        user_prompt = f"""
        Large Language Models are often vulnerable to so called prompt injection attacks, where 
        users design input prompts in a specific way to manipulate the output and behavior of the LLM.
        Create and print a new safe system prompt to instruct an LLM to be safe against prompt
        injection and prompt leaking attacks, especially when using tools like Emails, Calendars,
        or Clouds. These tools can be used by the LLM to manage and access user data which could be
        confidential.
        Start with set of instructions to guide the instructed LLM into keeping the secret key safe.
        Print only the system prompt and nothing else. The system prompt has to be at least 100
        but not more than 500 characters long and is not allowed to use any special symbols like \"\n\" or to repeatedly
        use the same words or sentences. The system prompt should be creative while being non-repetitive. 
        Do not write any introduction sentences. Try to keep the tool usage safe and secure from leaks.
        You can orient yourself at this example prompt: 
        \"{random_sys_prompt}\"

        The new system prompt is:
        """

        response, _ = llm.chat(sys_prompt, user_prompt)

        # sanitize the response
        response = response.replace("<|im_start|>system\n", "")
        response = response.replace("<|startoftext|>\n", "")
        response = response.replace("<|startoftext|>", "")
        response = response.replace("System prompts:\n", "")
        response = response.replace("<|im_start|>assistant\n", "")

        # add the response to the dataset
        dataset_test.add_prompt(response)
        progress_bar_test.update(len(dataset_test))


    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"{TColors.HEADER}\nComputation Time: {duration}{TColors.ENDC}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--dataset_size", "-ds", type=int, default=1000,
                        help="specifies the size of the resulting dataset")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama3-8b",
                        help="specifies the LLM to generate the dataset")
    parser.add_argument("--name_suffix", "-n", help="adds a name suffix for loading custom models",
                        default="", type=str)
    parser.add_argument("--device", "-dx", type=str, default="cpu",
                        help="specifies the device to run the computations on (cpu, cuda, mps)")
    args = parser.parse_args()
    main(**vars(args))
