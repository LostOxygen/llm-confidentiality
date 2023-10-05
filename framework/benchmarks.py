"""library for the benchmark helper functions"""
from typing import Type, Tuple

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

BENCHMARK_LIST = ["hellaswag"]


def get_benchmark_data(dataset_name: str) -> Type[DataLoader]:
    """
    Benchmark helper function which provides handy ways for initializing different datasets and 
    retreive their data in a convenient way (e.g. question/answer pairs in generators)
    """
    dataset: Type[Dataset] = None

    match dataset_name:
        case "hellaswag":
            dataset = load_dataset("Rowan/hellaswag", split="test")

        case _:
            raise ValueError(f"Unknown benchmark name: {dataset_name}")

    return DataLoader(dataset, batch_size=1, shuffle=True)


def format_benchmark_data(benchmark_name: str, data: dict) -> Tuple[str, str]:
    """
    Helper function which takes the data from the benchmark dataset and formats it into a
    suitable input string for the LLM.
    """
    user_prompt: str = ""
    system_prompt: str = ""

    match benchmark_name:
        case "hellaswag":
            system_prompt = """You are a large language model beeing tested on it's ability to
            finishing sentences. You get the first part of a sentence and have to choose the
            correct ending from a list of options. You are only allowed to choose one option and
            to only answer with the number of the option and nothing more.

            Here is the first part of the sentence:
            """

            user_prompt = f"""{data["ctx"]}
            The following options are available:
            1. {data["endings"][0]}
            2. {data["endings"][1]}
            3. {data["endings"][2]}
            4. {data["endings"][3]}

            Tell me the number of the correct ending.
            """

        case _:
            raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    return user_prompt, system_prompt
