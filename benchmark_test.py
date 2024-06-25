"""Langchain benchmark test script"""
import os
import sys
from pathlib import Path
import uuid
import datetime
import psutil
import getpass
import argparse

from huggingface_hub import login
import torch
from langsmith.client import Client
from langchain.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_benchmarks import (
    __version__,
    clone_public_dataset,
    registry,
)

from framework.colors import TColors
from framework.benchmark_agents import AgentFactory

if not os.path.isdir(str(Path.home() / "data")):
    os.mkdir(str(Path.home() / "data"))
os.environ["TRANSFORMERS_CACHE"] = str(Path.home() / "data")

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
    # and again for huggingfce
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
    # model: Type[LLM] = LLM(
    #     llm_type=llm_type,
    #     temperature=temperature,
    #     llm_suffix="",
    #     device=device
    # )
    model = ChatOllama(model="llama3", format="json", temperature=0)

    system = """
        Respond to the human as helpfully and accurately as possible. You have access to the following tools:

        {tools}

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}
        ```

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{
        "action": "Final Answer",
        "action_input": "Final response to human"
        }}

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
    
        {agent_scratchpad}
    """

    # Create prompts for the agents
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),  # Populated from task.instructions automatically
            #MessagesPlaceholder(variable_name="agent_scratchpad"),  # Workspace for the agent
        ]
    )

    # experiment_uuid = "gewürzgurke1337"
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
            prompt,
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
    parser.add_argument("--llm_type", "-llm", type=str, default="llama3-8b",
                        help="specifies the opponent LLM type")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="specifies the opponent LLM temperature")
    parser.add_argument("--device", "-dx", type=str, default="cpu",
                        help="specifies the device to run the computations on (cpu, cuda, mps)")
    args = parser.parse_args()
    main(**vars(args))
