"""main hook to start the LLaMA2 finetuning"""
# pylint: disable=inconsistent-quotes
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
from pathlib import Path
import sys
import time
import datetime
import getpass
import psutil
import argparse
from typing import (
    Final,
)

import torch
from huggingface_hub import login
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
#from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

from framework.colors import TColors
from framework.dataset import (
    ToolUseDataset,
    DatasetState
)

os.environ["TRANSFORMERS_CACHE"] = str(Path.home() / "data")
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_PROJECT"]="llm-finetuning"

# number of attack samples per attack type and main iteration
DATA_PATH: Final[str] = "./datasets/tool_use_train.json"
OUTPUT_DIR: Final[str] = "./finetuned_models/"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


CONFIG: Final[dict] = {
    "lora": {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "r": 16,
    },
    "training": {
        "output_dir": OUTPUT_DIR,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "logging_steps": 10,
        "max_steps": 1000,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "lr_scheduler_type": "linear",
    },
    "trainer": {
        "max_seq_length": 4096,
        "dataset_text_field": "prompts", # this is the training field name
        "packing": True,
        "dataset_num_proc": 2,
        "logging_steps": 1,
        "seed": 1337,
    }
}

def main(
        llm_type: str,
        iterations: int,
        name_suffix: str,
    ) -> None:
    """
    Main function to start the LLM finetuning.

    Parameters:
        llm_type: str - specifies the LLM type to finetune
        iterations: int - specifies the number of iterations to finetune the LLM
        name_suffix: str - specifies a name suffix for the finetuned model

    Returns:
        None
    """
    start = time.perf_counter()  # start timer
    # paste the Huggingface token into the hf_token.txt file and put into the root directory
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
        if llm_type in ["llama2", "llama2-7b", "llama2-13b", "llama2-70b"]:
            sys.exit(1)

    # set the devices correctly
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print(f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} " \
              f"{TColors.ENDC}is not available. Setting device to CPU instead.")
        device = torch.device("cpu")

    # setting the suffixes
    suffix: str = "finetuned"
    name_suffix: str = "-"+name_suffix if name_suffix != "" else ""
    # combine the finale output save name
    save_name: str = llm_type + "-" + suffix + name_suffix

    # update the default config
    CONFIG["training"]["max_steps"] = iterations

    # print system information
    print("\n"+f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-23))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if (device == "cuda" or torch.device("cuda")) and torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    elif (device == "mps" or torch.device("mps")) and torch.backends.mps.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    else:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    print(f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-14))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}LLM{TColors.ENDC}: {llm_type}")

    # print the finetuning parameters
    print(f"## {TColors.HEADER}{TColors.BOLD}{TColors.UNDERLINE}Finetuning Parameters " \
          f"{TColors.ENDC}" + "#"*int(os.get_terminal_size().columns-25))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}lora_alpha{TColors.ENDC}: " \
          f"{CONFIG['lora']['lora_alpha']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}lora_dropout{TColors.ENDC}: " \
          f"{CONFIG['lora']['lora_dropout']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}r-value{TColors.ENDC}: " \
          f"{CONFIG['lora']['r']}")
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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}num_train_epochs{TColors.ENDC}: " \
          f"{CONFIG['training']['num_train_epochs']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}optim{TColors.ENDC}: " \
            f"{CONFIG['training']['optim']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}weight_decay{TColors.ENDC}: " \
            f"{CONFIG['training']['weight_decay']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}warmup_steps{TColors.ENDC}: " \
            f"{CONFIG['training']['warmup_steps']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}lr_scheduler_type{TColors.ENDC}: " \
            f"{CONFIG['training']['lr_scheduler_type']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}seed{TColors.ENDC}: " \
            f"{CONFIG['trainer']['seed']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}output_dir{TColors.ENDC}: " \
            f"{CONFIG['training']['output_dir']}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}dataset_num_proc{TColors.ENDC}: " \
            f"{CONFIG['trainer']['dataset_num_proc']}")
    print("#"*os.get_terminal_size().columns+"\n")

    # load the LLM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )
    # convert the model to PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora"]["r"],
        lora_alpha=CONFIG["lora"]["lora_alpha"],
        lora_dropout=CONFIG["lora"]["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"
            ],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
    )
    # apply chat template to the tokenizer
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    #     chat_template="llama",
    #     map_eos_token=True,
    # )

    # load the dataset
    assert os.path.isfile(DATA_PATH), f"{TColors.FAIL}Couldn't find dataset.{TColors.ENDC}"
    prompt_dataset = ToolUseDataset(state=DatasetState.TRAIN)
    dataset = Dataset.from_dict({"prompts": prompt_dataset.get_whole_dataset_as_list()})

    print(f">> {TColors.OKBLUE}Normal Finetuning for {iterations} steps{TColors.ENDC}")

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=CONFIG["trainer"]["dataset_text_field"],
        max_seq_length=CONFIG["trainer"]["max_seq_length"],
        dataset_num_proc=CONFIG["trainer"]["dataset_num_proc"],
        packing=CONFIG["trainer"]["packing"],
        args=TrainingArguments(
            learning_rate=CONFIG["training"]["learning_rate"],
            lr_scheduler_type=CONFIG["training"]["lr_scheduler_type"],
            per_device_train_batch_size=CONFIG["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=CONFIG["training"]["gradient_accumulation_steps"],
            num_train_epochs=CONFIG["training"]["num_train_epochs"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=CONFIG["training"]["logging_steps"],
            optim=CONFIG["training"]["optim"],
            weight_decay=CONFIG["training"]["weight_decay"],
            warmup_steps=CONFIG["training"]["warmup_steps"],
            output_dir=OUTPUT_DIR,
            seed=CONFIG["trainer"]["seed"],
        ),
    )

    trainer.train()

    # save the model
    # model.save_pretrained_merged(
    #     OUTPUT_DIR+save_name,
    #     tokenizer,
    #     save_method="merged_16bit"
    # )

    # saving to GGUF for ollama
    model.save_pretrained_gguf(
        OUTPUT_DIR+save_name,
        tokenizer,
        # https://docs.unsloth.ai/basics/saving-models/saving-to-gguf
        quantization_method=["q4_k_m"],
    )

    print(f"{TColors.OKGREEN}Finetuning finished.{TColors.ENDC}")
    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"{TColors.HEADER}Computation Time: {duration}{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama3-8b",
                        help="specifies the opponent LLM type")
    parser.add_argument("--iterations", "-i", type=int, default=1000,
                        help="specifies the number of iterations to finetune the LLM")
    parser.add_argument("--name_suffix", "-n", help="adds a name suffix for the finetuned model",
                        default="", type=str)
    args = parser.parse_args()
    main(**vars(args))
