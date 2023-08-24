"""library of utility functions"""
import os
import datetime

LOG_PATH: str = "./logs/"


def log_conversation(llm_name: str, attack_name: str, defense_name: str, level: int,
                     prompt: str, response: str, success: bool) -> None:
    """
    Logs the conversation between the user and the opponent LLM into 
    the {llm_name}_chat_logs.txt file.

    Parameters:
        llm_name: str - the type of the opponent LLM
        attack_name: str - the name of the attack
        defense_name: str - the name of the defense
        level: int - the level of the attack
        prompt: str - the prompt sent to the opponent LLM
        response: str - the response of the opponent LLM
        success: bool - whether the response contains the secret key or not

    Returns:
        None
    """
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    file_name = LOG_PATH + f"{llm_name}_chat_logs.txt"

    with open(file=file_name, mode="a", encoding="utf-8") as f:
        f.write("\n"+"#"*100)
        f.write(f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n")
        f.write(f">>LLM Type: {llm_name}\n")
        f.write(f">>Attack: {attack_name}\n")
        f.write(f">>Defense: {defense_name}\n")
        f.write(f">>Level: {level}\n")
        f.write(f">>Success: {success}\n")
        f.write(f">>Prompt: \n{prompt}\n")
        f.write(f">>Response: \n{response}\n")
        f.write("\n"+"#"*100)


def log_results(llm_name: str, defense_name: str, success_dict: dict, iters: int) -> None:
    """
    Logs the final attack/defense results into the {llm_name}_result_logs.txt file.

    Parameters:
        llm_name: str - the type of the opponent LLM
        attack_name: str - the name of the attack
        defense_name: str - the name of the defense
        success_dict: dict - the dictionary containing the attack/defense results
        iters: int - the number of iterations for the attacks

    Returns:
        None
    """

    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    file_name = LOG_PATH + f"{llm_name}_result_logs.txt"

    with open(file=file_name, mode="a", encoding="utf-8") as f:
        f.write("\n"+"#"*100)
        f.write(f"\n>>Time: {str(datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p'))}\n")
        f.write(f">>LLM Type: {llm_name}\n")
        f.write(f">>Defense: {defense_name}\n")
        f.write(">>Attack Results: \n\n")

        total_successes: int = 0
        total_iterations: int = 0
        for attack_name, successes in success_dict.items():
            total_successes += successes
            total_iterations += iters
            percentage = round(successes/iters*100, 2)
            f.write(f">>Attack: {attack_name} - Successes: {successes}/{iters} -> {percentage}%\n")

        total_percentage = round(total_successes/total_iterations*100, 2)
        f.write(f">>Total Successes: {total_successes}/{total_iterations} -> {total_percentage}%\n")
        f.write("\n"+"#"*100)
