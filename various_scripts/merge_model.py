"""helper script to merge finetuned models with it's original model"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import datetime
import socket
import argparse
from typing import Final

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel

from various_scripts.colors import TColors

os.environ["TRANSFORMERS_CACHE"] = "/data/"
OUTPUT_DIR: Final[str] = "./merged_models/"

def main(base_llm: str, finetuned_model: str) -> None:
    """
    Merges the finetuned model with the original model and saves it to the
    OUTPUT_DIR directory
    
    Args:
        base_llm (str): the base LLM type
        finetuned_model (str): the path of the adapters to merge

    Returns:
        None
    """
    adapter_name = finetuned_model.split("/")[-1]
    save_path = os.path.join(OUTPUT_DIR, adapter_name) + "_merged"

    # print system information
    print("\n"+"#"*os.get_terminal_size().columns)
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: cpu")
    if torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Base-LLM{TColors.ENDC}: {base_llm}")

    print(f"## {TColors.OKBLUE}{TColors.BOLD}Adapter/PEFT Model:{TColors.ENDC}: {adapter_name}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Save Path:{TColors.ENDC}: {save_path}")


    base_tokenizer = AutoTokenizer.from_pretrained(
        base_llm,
        use_fast=False
    )
    base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_llm,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )

    merged_model = PeftModel.from_pretrained(
        base_model, # base model
        finetuned_model, # adapter/peft model weights
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=os.environ["TRANSFORMERS_CACHE"]
    )

    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(
        save_path,
        safe_serialization=True,
        save_adapter=True,
        save_config=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--base_llm", "-llm", type=str, default="llama2-7b",
                        help="specifies the base LLM type", required=True)
    parser.add_argument("--finetuned_model", "-finetuned", type=str, required=True,
                        help="specifies the path of the adapters to merge")
    args = parser.parse_args()
    main(**vars(args))
