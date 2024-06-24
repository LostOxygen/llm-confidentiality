"""Langchain benchmark test script"""
import os
import uuid
import datetime
import psutil
from typing import Type
from getpass import getpass
import argparse

import torch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith.client import Client
from langchain_benchmarks import (
    __version__,
    clone_public_dataset,
    registry,
)
from langchain_benchmarks.rate_limiting import RateLimiter

from framework.colors import TColors
from framework.llm import LLM
from framework.benchmark_agents import AgentFactory

def main(
        llm_type: str,
        temperature: float,
        device: str,
) -> None:
    """
    Main hook for benchmarking llms via langchain

    Parameters:
        llm_type: str - specifies which llm to benchmark
        temperature: float - specifices the used temperature for llm generation
        device: str - specifies the used computation device (cpu, cuda, mps)

    Returns:
        None
    
    """

    # paste the OpenAI key into the key.txt file and put into the root directory
    try:
        with open(file="key.txt", mode="r", encoding="utf-8") as f:
            key = f.read().replace("\n", "")
            assert key != "", f"{TColors.FAIL}Key is empty.{TColors.ENDC}"

            os.environ["OPENAI_API_KEY"] = key
            print(f"{TColors.OKGREEN}OpenAI API key loaded.{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your OpenAI API key into the key.txt "
                f"file and put it into the root directory.{TColors.ENDC}")

    # now same for langsmith api key
    try:
        with open(file="langsmith_key.txt", mode="r", encoding="utf-8") as f:
            key = f.read().replace("\n", "")
            assert key != "", f"{TColors.FAIL}Key is empty.{TColors.ENDC}"

            os.environ["LANGCHAIN_API_KEY"] = key
            print(f"{TColors.OKGREEN}Langsmith API key loaded.{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your Langsmith API key into the langsmith_key.txt "
                f"file and put it into the root directory.{TColors.ENDC}")

    print("\n"+f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information" + \
            f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-23))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    elif device == "mps" and torch.backends.mps.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    else:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    print(f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-14))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}LLM{TColors.ENDC}: " \
          f"{TColors.HEADER}{llm_type}{TColors.ENDC}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Temperature{TColors.ENDC}: {temperature}")
    print("#"*os.get_terminal_size().columns+"\n")

    # create the LLM
    model: Type[LLM] = LLM(
        llm_type=llm_type,
        temperature=temperature,
        llm_suffix="",
        device=device
    )

    # Create prompts for the agents
    # with_system_message_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "{instructions}"),
    #         ("human", "{question}"),  # Populated from task.instructions automatically
    #         MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
    #     ]
    # )

    # experiment_uuid = "gewürzgurke1337"  # Or generate random using uuid.uuid4().hex[:4]
    experiment_uuid = uuid.uuid4().hex[:4]

    client = Client()  # Launch langsmith client for cloning datasets
    today = datetime.date.today().isoformat()


    for task in registry.tasks:
        if task.type != "ToolUsageTask":
            continue

        # This is a small test dataset that can be used to verify
        # that everything is set up correctly prior to running over
        # all results. We may remove it in the future.
        if task.name == "Multiverse Math (Tiny)":
            continue

        dataset_name = task.name + f" ({today})"
        clone_public_dataset(task.dataset_id, dataset_name=dataset_name)

        print()
        print(f"Benchmarking {task.name} with model: {llm_type}")
        eval_config = task.get_eval_config()

        agent_factory = AgentFactory(
            task,
            model,
        )

        client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=agent_factory,
            evaluation=eval_config,
            verbose=False,
            project_name=f"{llm_type}-{task.name}-{today}-{experiment_uuid}",
            concurrency_level=5,
            project_metadata={
                "model": llm_type,
                "id": experiment_uuid,
                "task": task.name,
                "date": today,
                "langchain_benchmarks_version": __version__,
            },
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama2-7b",
                        help="specifies the opponent LLM type")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="specifies the opponent LLM temperature")
    parser.add_argument("--device", "-dx", type=str, default="cpu",
                        help="specifies the device to run the computations on (cpu, cuda, mps)")
    args = parser.parse_args()
    main(**vars(args))
