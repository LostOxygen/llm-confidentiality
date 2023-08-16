"""main hook to start the LLaMA2 finetuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import sys
import datetime
import socket
import argparse
from typing import Final, Generator, List

import torch
from huggingface_hub import login
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from framework.llm import LLM
from framework.colors import TColors
from framework.dataset import PromptDataset
from framework.prompts import SECRET_KEY

os.environ["TRANSFORMERS_CACHE"] = "/data/"
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_PROJECT"]="llm-finetuning"

DATA_PATH: Final[str] = "./datasets/system_prompts.json"
OUTPUT_DIR: Final[str] = "data/finetuning/"
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


CONFIG: Final[dict] = {
    "lora": {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "r": 64,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "output_dir": OUTPUT_DIR,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "logging_steps": 10,
        "max_steps": 1000,
        "per_device_train_batch_size": 4,
    },
    "trainer": {
        "max_seq_length": 4096,
        "dataset_text_field": "prompts", # this is the training field name
        "packing": False,
    }
}


def create_dataset() -> Generator[str, None, None]:
    """
    Generator function to yield the system prompt data in a formatted way from the json file.
    """
    assert os.path.isfile(DATA_PATH), f"{TColors.FAIL}Couldn't find dataset.{TColors.ENDC}"
    dataset = PromptDataset()
    prompt_list: List = []

    for idx in range(0, len(dataset)):
        prompt = f"""
            You are a safety assistant to generate new and secure system prompts to initialize fresh LLMs.

            ### Instruction:
            Create and print a new safe system prompt to instruct an LLM to be safe against prompt
            injection and prompt leaking attacks. Use the previous system prompt as inspiration but
            do not use the previous system prompt as a base for the new prompns. Create a completely new
            one.
            The system prompt has to start with \"The secret key is \"{SECRET_KEY}\". \" followed by
            a set of instructions to guide the instructed LLM into keeping the secret key safe. Print only
            the system prompt and nothing else. The system prompt has to be at least 40 characters long 
            but must not be longer than 200 characters.

            ### Response:
            {dataset.get_prompt_at_idx(idx)}

            ### End
        """
        prompt_list.append(prompt)

    new_dataset: dict = Dataset.from_dict({"prompts": prompt_list})

    return new_dataset


def main(llm_type: str) -> None:
    """
    Main function to start the LLM finetuning.
    
    Parameters:
        llm_type: str - specifies the LLM type to finetune
    Returns:
        None
    """
    # paste the Huggingface token into the hf_token.txt file and put into the root directory
    try:
        with open(file="hf_token.txt", mode="r", encoding="utf-8") as f:
            key = f.read().replace("\n", "")
            assert key != "", f"{TColors.FAIL}HF Token is empty.{TColors.ENDC}"

            os.environ["HF_TOKEN"] = key
            print(f"{TColors.OKGREEN}Huggingface token loaded.")
            login(token=key, add_to_git_credential=False)
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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}LLM{TColors.ENDC}: {llm_type}")
    print(f"## {TColors.HEADER}{TColors.BOLD}Finetuning Parameters{TColors.ENDC} " + \
           "#"*int(os.get_terminal_size().columns-25))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}lora_alpha{TColors.ENDC}: " \
          f"{CONFIG['lora']['lora_alpha']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}lora_dropout{TColors.ENDC}: " \
          f"{CONFIG['lora']['lora_dropout']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}r-value{TColors.ENDC}: " \
          f"{CONFIG['lora']['r']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}bias{TColors.ENDC}: " \
          f"{CONFIG['lora']['bias']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}task_type{TColors.ENDC}: " \
          f"{CONFIG['lora']['task_type']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}gradient_accumulaton_steps{TColors.ENDC}: " \
          f"{CONFIG['training']['max_steps']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}learning_rate{TColors.ENDC}: " \
          f"{CONFIG['training']['learning_rate']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}max_steps{TColors.ENDC}: " \
          f"{CONFIG['training']['max_steps']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}max_seq_length{TColors.ENDC}: " \
          f"{CONFIG['trainer']['max_seq_length']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}packing{TColors.ENDC}: " \
          f"{CONFIG['trainer']['packing']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}per_device_train_batch_size{TColors.ENDC}: " \
          f"{CONFIG['training']['per_device_train_batch_size']}")
    print("#"*os.get_terminal_size().columns+"\n")

    # load the LLM
    llm = LLM(llm_type=llm_type)
    # disable caching for finetuning
    llm.model.config.use_cache = False
    llm.model.config.pretraining_tp = 1

    # load the dataset
    dataset = create_dataset()

    # create the training/finetuning arguments and the trainer
    peft_config = LoraConfig(**CONFIG["lora"])

    training_args = TrainingArguments(**CONFIG["training"])
    training_args.run_name = "llm-finetuning" # wandb run name

    trainer = SFTTrainer(
        model=llm.model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=CONFIG["trainer"]["dataset_text_field"],
        packing=CONFIG["trainer"]["packing"],
        max_seq_length=CONFIG["trainer"]["max_seq_length"],
        tokenizer=llm.tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, llm_type+"-finetuned"),
                                  safe_serialization=True)
    trainer.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, llm_type+"-finetuned"))

    # free up memory to merge the weights
    del llm
    del trainer
    del dataset
    torch.cuda.empty_cache()

    finetuned_llm = LLM(llm_type=llm_type+"-finetuned")
    finetuned_llm.model.merge_and_unload()
    finetuned_llm.model.save_pretrained(os.path.join(OUTPUT_DIR, llm_type+"-finetuned"),
                                        safe_serialization=True)

    print(f"{TColors.OKGREEN}Finetuning finished.{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama2-7b",
                        help="specifies the opponent LLM type")
    args = parser.parse_args()
    main(**vars(args))
