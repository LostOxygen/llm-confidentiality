"""Langchain benchmark test script"""
import os
import uuid
import datetime
from getpass import getpass

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langsmith.client import Client
from langchain_benchmarks.tool_usage.agents import StandardAgentFactory
from langchain_benchmarks import (
    __version__,
    clone_public_dataset,
    registry,
)
from langchain_benchmarks.rate_limiting import RateLimiter

from framework.colors import TColors

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


# This is just the default list below
required_env_vars = [
    "OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
]
for var in required_env_vars:
    if var not in os.environ:
        os.environ[var] = getpass(f"Provide the required {var}")

tests = [
    (
        "gpt-4-turbo-2024-04-09",
        ChatOpenAI(model="gpt-4-turbo-2024-04-09", temperature=0),
    ),
]

# Create prompts for the agents
# Using two prompts because some chat models do not support SystemMessage.
without_system_message_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "{instructions}\n{question}",
        ),  # Populated from task.instructions automatically
        MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
    ]
)

with_system_message_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instructions}"),
        ("human", "{question}"),  # Populated from task.instructions automatically
        MessagesPlaceholder("agent_scratchpad"),  # Workspace for the agent
    ]
)

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

    for model_name, model in tests:
        if model_name.startswith("gemini"):
            # google models don't use system prompt
            prompt = without_system_message_prompt
            rate_limiter = RateLimiter(requests_per_second=0.1)
        else:
            prompt = with_system_message_prompt
            rate_limiter = RateLimiter(requests_per_second=1)
        print()
        print(f"Benchmarking {task.name} with model: {model_name}")
        eval_config = task.get_eval_config()

        agent_factory = StandardAgentFactory(
            task, model, prompt, rate_limiter=rate_limiter
        )

        client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=agent_factory,
            evaluation=eval_config,
            verbose=False,
            project_name=f"{model_name}-{task.name}-{today}-{experiment_uuid}",
            concurrency_level=5,
            project_metadata={
                "model": model_name,
                "id": experiment_uuid,
                "task": task.name,
                "date": today,
                "langchain_benchmarks_version": __version__,
            },
        )
