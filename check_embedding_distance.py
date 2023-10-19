"""helper script to evaluate the embedding space distance of each attack dataset"""
import os
import argparse
import datetime
import socket

import torch
from sentence_transformers import SentenceTransformer

from framework.dataset import PromptDataset, DatasetState
from framework.colors import TColors

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
    model = SentenceTransformer("all-mpnet-base-v2 ")
    prompt = dataset.get_random_prompt()
    embedding = model.encode(prompt)

    print(f"{TColors.HEADER}Prompt{TColors.ENDC}:", prompt)
    print(f"{TColors.HEADER}Embedding{TColors.ENDC}:", embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Space Distance")
    args = parser.parse_args()
    main(**vars(args))
