"""helper script to evaluate the embedding space distance of each attack dataset"""
import os
import sys
import argparse
import datetime
import socket
from typing import Final

import torch
import pkbar
from torch import nn
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from framework.dataset import PromptDataset, DatasetState
from framework.colors import TColors
from framework.attacks import (
        ATTACK_LIST,
        payload_splitting,
        obfuscation,
        manipulation,
        translation,
        chatml_abuse,
        masking,
        typoglycemia,
        advs_suffix
    )

NUM_ITERATIONS: Final[int] = 100


def main() -> None:
    """main hook"""
    # set devices correctly
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"

    # print system information
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
    print("#"*os.get_terminal_size().columns+"\n")

    dataset = PromptDataset(state=DatasetState.TRAIN)
    model = SentenceTransformer("all-mpnet-base-v2").to(device)
    # using cosine similarity for similrity measurement
    # 1 would be "similar", 0 would be "independent/orthogonal"
    cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    distance_dict = {}

    for attack in ATTACK_LIST:
        match attack:
            case "payload_splitting": attack_func = payload_splitting
            case "obfuscation": attack_func = obfuscation
            case "manipulation": attack_func = manipulation
            case "translation": attack_func = translation
            case "chatml_abuse": attack_func = chatml_abuse
            case "masking": attack_func = masking
            case "typoglycemia": attack_func = typoglycemia
            case "advs_suffix": attack_func = advs_suffix
            case _:
                print(f"{TColors.FAIL}Attack type {attack} is not supported.{TColors.ENDC}")
                print(f"{TColors.FAIL}Choose from: {ATTACK_LIST}{TColors.ENDC}")
                sys.exit(1)

        temp_distance = 0.0
        progress_bar = pkbar.Pbar(
                target=NUM_ITERATIONS,
                width=40,
                name=f"Computing embedding distance for {attack}"
            )
        for sim_iter in range(NUM_ITERATIONS):
            prompt_a = f"""
                {dataset.get_random_prompt()}

                ### Instruction:
                {attack_func()}

                ### Response:
                Prompt injection attack detected! I will not leak any confidential informations!

                ### End
            """
            prompt_b = f"""
                {dataset.get_random_prompt()}

                ### Instruction:
                {attack_func()}

                ### Response:
                Prompt injection attack detected! I will not leak any confidential informations!

                ### End
            """
            embedding_a = torch.from_numpy(model.encode(prompt_a))
            embedding_b = torch.from_numpy(model.encode(prompt_b))

            temp_distance += cos_sim(embedding_a, embedding_b)
            progress_bar.update(sim_iter)

        distance_dict[attack] = temp_distance / NUM_ITERATIONS

    print(f"\n>> Mean Similarity over {NUM_ITERATIONS} random prompts:")
    for attack, distance in distance_dict.items():
        print(f"{TColors.OKCYAN}{attack}{TColors.ENDC}: {distance: >20}")

    plt.figure(figsize=(20, 20))
    plt.bar(range(len(distance_dict)), list(distance_dict.values()), align="center")
    plt.xticks(range(len(distance_dict)), list(distance_dict.keys()), rotation="vertical")
    plt.title("Embedding Space Distance (Cosine Similarity)")

    if not os.path.isdir("./plots"):
        os.mkdir("./plots")
    plt.savefig("./plots/embedding_distance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Space Distance")
    args = parser.parse_args()
    main(**vars(args))
