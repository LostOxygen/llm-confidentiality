"""main hook to start the llm evaluation"""
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

from framework.strategy import BenchmarkStrategy
from framework.benchmarks import BENCHMARK_LIST
from framework.colors import TColors
from framework.utils import log_benchmark


if not os.path.isdir("/data/"):
    os.mkdir("/data/")
os.environ["TRANSFORMERS_CACHE"] = "/data/"


def main(
        benchmarks: List[str],
        llm_type: str,
        temperature: float,
        name_suffix: str,
        iterations: int
    ) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        benchmarks: List[str] - specifies the benchmark types
        llm_type: str - specifies the opponent LLM type
        temperature: float - specifies the opponent LLM temperature to control randomness
        name_suffix: str - adds a name suffix for loading custom models
        iterations: int - specifies the number of testing iterations

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
        if llm_type in ["gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-4"]:
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

    if "all" in benchmarks:
        benchmarks = BENCHMARK_LIST

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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Benchmarks{TColors.ENDC}: {benchmarks}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Opponent LLM{TColors.ENDC}: " \
          f"{TColors.HEADER}{llm_type}{TColors.OKCYAN}{name_suffix}{TColors.ENDC}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Temperature{TColors.ENDC}: {temperature}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Test Iterations{TColors.ENDC}: {iterations}")
    print("#"*os.get_terminal_size().columns+"\n")

    total_successrates: dict[int] = {f"{benchmark}" : 0 for benchmark in benchmarks}

    # initialize the strategy
    strategy = BenchmarkStrategy(
                benchmark_name=benchmarks[0],
                llm_type=llm_type,
                llm_suffix=name_suffix,
                temperature=temperature,
                test_iterations=iterations,
            )

    for benchmark in benchmarks:
        # load the benchmark into the strategy
        if strategy.benchmark_name != benchmark:
            strategy.update_benchmark(benchmark)

        # run the benchmark
        total_successrates[benchmark] = strategy.execute()
        torch.cuda.empty_cache()

        # print and log the results
        print(f"{TColors.OKBLUE}{TColors.BOLD}>> Benchmark Results:{TColors.ENDC}")
        for benchmark, successrate in total_successrates.items():
            print(f"Benchmark: {TColors.OKCYAN}{benchmark}{TColors.ENDC} - " \
                  f"Score: {successrate}%")

        log_benchmark(
                llm_name=llm_type+name_suffix,
                success_dict=total_successrates,
                iters=iterations
            )

    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"{TColors.HEADER}Computation Time: {duration}{TColors.ENDC}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--benchmarks", "-b", type=str, default=["hellaswag"],
                        help="specifies the benchmark types", nargs="+")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama2-7b",
                        help="specifies the LLM type which gets tested")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="specifies the opponent LLM temperature")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                        help="specifies the number of testing iterations")
    parser.add_argument("--name_suffix", "-n", help="adds a name suffix for loading custom models",
                        default="", type=str)
    args = parser.parse_args()
    main(**vars(args))
