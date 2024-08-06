"""library of utility functions"""
# pylint: disable=inconsistent-quotes
import os
import datetime
import random
from typing import Optional

LOG_PATH: str = "./logs/"


def log_conversation(
        llm_name: str,
        attack_name: str,
        defense_name: str,
        iteration: int,
        prompt: str,
        sys_prompt: str,
        response: str,
        success: bool,
        secret_key: str,
        intermediate_steps: Optional[str] = "",
        scenario: Optional[str] = None,
        overwrite: Optional[bool] = False,
    ) -> None:
    """
    Logs the conversation between the user and the opponent LLM into 
    the {llm_name}_chat_logs.txt file.

    Parameters:
        llm_name: str - the type of the opponent LLM
        attack_name: str - the name of the attack
        defense_name: str - the name of the defense
        iteration: int - the iteration of the attack
        prompt: str - the prompt sent to the opponent LLM
        sys_prompt: str - the system prompt to initialize the opponent LLM
        response: str - the response of the opponent LLM
        success: bool - whether the response contains the secret key or not
        secret_key: str - the secret key
        intermediate_steps: str - the intermediate steps of the llm for tool usage
        scenario: str - the scenario of the attack (if the tool attack strategy is used)
        overwrite: bool - whether to overwrite the file or not

    Returns:
        None
    """
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    file_name = LOG_PATH + f"{llm_name}_chat_logs.txt"

    if overwrite:
        mode = "w"
    else:
        mode = "a"

    with open(file=file_name, mode=mode, encoding="utf-8") as f:
        f.write("\n"+"#"*100)
        f.write(f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n")
        f.write(f">>LLM Type: {llm_name}\n")
        if scenario:
            f.write(f">>Scenario: {scenario}\n")
        f.write(f">>Attack: {attack_name}\n")
        f.write(f">>Defense: {defense_name}\n")
        f.write(f">>iteration: {iteration}\n")
        f.write(f">>Success: {success}\n")
        f.write(f">>Key: {secret_key}\n")
        f.write(f">>System Prompt: \n{sys_prompt}\n")
        f.write(f">>Prompt: \n{prompt}\n")
        f.write(f"Intermediate Steps: \n{intermediate_steps}\n")
        f.write(f">>Response: \n{response}\n")
        f.write("\n"+"#"*100)


def log_benchmark_conversation(
        llm_name: str,
        benchmark_name: str,
        iteration: int,
        prompt: str,
        sys_prompt: str,
        response: str,
        success: bool
    ) -> None:
    """
    Logs the conversation between the user and the opponent LLM into 
    the {llm_name}_benchmark_chat_logs.txt file.

    Parameters:
        llm_name: str - the type of the opponent LLM
        benchmark: str - the name of the benchmark
        iteration: int - the iteration of the attack
        prompt: str - the prompt sent to the opponent LLM
        sys_prompt: str - the system prompt to initialize the opponent LLM
        response: str - the response of the opponent LLM
        success: bool - whether the response was correct or not 

    Returns:
        None
    """
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    file_name = LOG_PATH + f"{llm_name}_benchmark_chat_logs.txt"

    with open(file=file_name, mode="a", encoding="utf-8") as f:
        f.write("\n"+"#"*100)
        f.write(f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n")
        f.write(f">>LLM Type: {llm_name}\n")
        f.write(f">>Benchmark: {benchmark_name}\n")
        f.write(f">>iteration: {iteration}\n")
        f.write(f">>Success: {success}\n")
        f.write(f">>System Prompt: \n{sys_prompt}\n")
        f.write(f">>Prompt: \n{prompt}\n")
        f.write(f">>Response: \n{response}\n")
        f.write("\n"+"#"*100)


def log_results(
        llm_name: str,
        defense_name: str,
        success_dict: dict,
        error_dict: dict,
        iters: int,
        overwrite: Optional[bool] = False,
        scenario: Optional[str] = None,
    ) -> None:
    """
    Logs the final attack/defense results into the {llm_name}_result_logs.txt file.

    Parameters:
        llm_name: str - the type of the opponent LLM
        defense_name: str - the name of the defense
        success_dict: dict - the dictionary containing the attack/defense results
        iters: int - the number of iterations for the attacks
        overwrite: bool - whether to overwrite the file or not
        scenario: str - the scenario of the attack (if the tool attack strategy is used)

    Returns:
        None
    """

    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    file_name = LOG_PATH + f"{llm_name}_result_logs.txt"

    if overwrite:
        mode = "w"
    else:
        mode = "a"

    with open(file=file_name, mode=mode, encoding="utf-8") as f:
        f.write("\n"+"#"*100)
        f.write(f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n")
        f.write(f">>LLM Type: {llm_name}\n")
        f.write(f">>Defense: {defense_name}\n")
        if scenario:
            f.write(f">>Scenario: {scenario}\n")
        f.write(">>Attack Results: \n\n")

        total_successes: int = 0
        total_iterations: int = 0
        for attack_name, successes, errors in zip(success_dict.items(), error_dict.values()):
            total_successes += successes
            # the iterations are without errors
            total_iterations += (iters - errors)
            percentage = round(successes/(iters-errors)*100, 2)

            f.write(f">>Attack: {attack_name} - Successes: " \
                    f"{successes}/{iters - errors} -> {percentage}%\n")

        total_percentage = round(total_successes/total_iterations*100, 2)
        f.write(f">>Total Successes: {total_successes}/{total_iterations} -> {total_percentage}%\n")
        f.write("\n"+"#"*100)


def log_benchmark(
        llm_name: str,
        success_dict: dict,
        iters: int
    ) -> None:
    """
    Logs the final benchmark results into the {llm_name}_benchmark_result_logs.txt file.

    Parameters:
        llm_name: str - the type of the opponent LLM
        success_dict: dict - the dictionary containing the benchmark results
        iters: int - the number of iterations for the attacks

    Returns:
        None
    """

    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    file_name = LOG_PATH + f"{llm_name}_benchmark_result_logs.txt"

    with open(file=file_name, mode="a", encoding="utf-8") as f:
        f.write("\n"+"#"*100)
        f.write(f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n")
        f.write(f">>LLM Type: {llm_name}\n")
        f.write(">>Benchmark Results: \n\n")

        total_successrate: int = 0
        total_iterations: int = 0
        for benchmark_name, successrate in success_dict.items():
            total_successrate += successrate
            total_iterations += iters
            f.write(f">>Becnhmark: {benchmark_name} - Score: {successrate}%\n")

        total_score = round(total_successrate/total_iterations, 2)
        f.write(f">>Total Score: {total_score}%\n")
        f.write("\n"+"#"*100)


def make_string_typoglycemiac(text: str) -> str:
    """
    Returns a string with the characters of each word randomly rearranged, 
    except for the first and last characters.

    Parameters:
        text: str - the input text

    Returns:
        str - the text with the characters of each word randomly rearranged
    """
    result = " ".join([word[0] + "".join(random.sample(word[1:-1], \
             len(word[1:-1]))) + word[-1] for word in text.split()])
    return result
