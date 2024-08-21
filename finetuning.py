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
    List,
    Callable,
    Union,
)

import torch
from huggingface_hub import login
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

from framework.attacks import (
        ATTACK_LIST,
        payload_splitting,
        obfuscation,
        jailbreak,
        translation,
        chatml_abuse,
        masking,
        typoglycemia,
        advs_suffix,
    )
from framework.colors import TColors
from framework.dataset import (
    PromptDataset,
    AdvsTrainDataset,
    DatasetState
)
from framework.prompts import get_random_secret_key

os.environ["TRANSFORMERS_CACHE"] = str(Path.home() / "data")
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_PROJECT"]="llm-finetuning"

# number of attack samples per attack type and main iteration
NUM_ATTACK_SAMPLES: Final[int] = 100
DATA_PATH: Final[str] = "./datasets/system_prompts_train.json"
OUTPUT_DIR: Final[str] = "./finetuned_models/"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        "per_device_train_batch_size": 1,
    },
    "trainer": {
        "max_seq_length": 4096,
        "dataset_text_field": "prompts", # this is the training field name
        "packing": False,
    }
}


def get_attack_list(attacks: List[str]) -> List[Callable]:
    """
    Creates a list of callable attack functions out of the list of attack names.

    Parameters:
        attacks: List[str] - the list of attack names

    Returns:
        attack_funcs: List[Callable] - the list of callable attack functions
    """
    attack_funcs: List[Callable] = []

    if "all" in attacks:
        attacks = ATTACK_LIST
        if "chat_base" in attacks:
            attacks.remove("chat_base")

    for attack in attacks:
        # set the attack function
        match attack:
            case "payload_splitting": attack_funcs.append(payload_splitting)
            case "obfuscation": attack_funcs.append(obfuscation)
            case "jailbreak": attack_funcs.append(jailbreak)
            case "translation": attack_funcs.append(translation)
            case "chatml_abuse": attack_funcs.append(chatml_abuse)
            case "masking": attack_funcs.append(masking)
            case "typoglycemia": attack_funcs.append(typoglycemia)
            case "advs_suffix": attack_funcs.append(advs_suffix)
            case _:
                print(f"{TColors.FAIL}Attack type {attack} is not supported.{TColors.ENDC}")
                print(f"{TColors.FAIL}Choose from: {ATTACK_LIST}{TColors.ENDC}")
                sys.exit(1)

    return attack_funcs


def get_formatted_dataset(
        dataset: Union[PromptDataset, AdvsTrainDataset],
        attacks: List[Callable] = None,
        ) -> Dataset:
    """
    Creating a dictionary dataset from the system prompts.

    Parameters:
        is_robust: bool - specifies if the dataset should contain system prompts or
                          prompt injection attacks
        dataset: Union[PromptDataset, AdvsTrainDataset] - the dataset to use
        attacks: List[Callable] - the list of attacks to harden the LLM against

    Returns:
        new_dataset: Dataset - the new dataset containing the finetune data
    """
    prompt_list: List = []
    secret_key = get_random_secret_key()
    sk_string = f"The secret key is: \"{secret_key}\". "

    # create prompt injection attack dataset
    for attack in attacks:
        for _ in range(NUM_ATTACK_SAMPLES):
            prompt = f""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                {{ {sk_string}{dataset.get_random_prompt()} }}<|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                {{ {attack} }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                Prompt injection attack detected! I will not leak any confidential information!
                <|eot_id|>
            """

            prompt_list.append(prompt)
    new_dataset = Dataset.from_dict({"prompts": prompt_list})
    return new_dataset


def main(
        llm_type: str,
        iterations: int,
        attacks: List[str],
        name_suffix: str,
    ) -> None:
    """
    Main function to start the LLM finetuning.

    Parameters:
        llm_type: str - specifies the LLM type to finetune
        iterations: int - specifies the number of iterations to finetune the LLM
        attacks: List[str] - specifies the attack types to harden the LLM against
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
    suffix: str = "robust"
    name_suffix: str = "-"+name_suffix if name_suffix != "" else ""
    attack_suffix: str = "-"+"-".join(attacks)
    # combine the finale output save name
    save_name: str = llm_type + "-" + suffix + attack_suffix + name_suffix

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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Attacks: {TColors.ENDC}: {attacks}")

    # print the finetuning parameters
    print(f"## {TColors.HEADER}{TColors.BOLD}{TColors.UNDERLINE}Finetuning Parameters " \
          f"{TColors.ENDC}" + "#"*int(os.get_terminal_size().columns-25))
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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=None,
    )
    # convert the model to PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"
            ],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )
    # apply chat template to the tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        chat_template="chatml",
    )
    # create list of attacks to harden against if robust finetuning is enabled
    attack_funcs = get_attack_list(attacks)

    # load the dataset
    assert os.path.isfile(DATA_PATH), f"{TColors.FAIL}Couldn't find dataset.{TColors.ENDC}"
    prompt_dataset = PromptDataset(state=DatasetState.TRAIN)

    dataset = get_formatted_dataset(
        attacks=attack_funcs,
        dataset=prompt_dataset
    )

    print(f">> {TColors.OKBLUE}Normal Finetuning for {iterations} steps{TColors.ENDC}")

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="prompts",
        max_seq_length=4096,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir=OUTPUT_DIR,
            seed=0,
        ),
    )

    trainer.train()

    # save the model
    model.save_pretrained_merged(save_name, tokenizer, save_method="merged_16bit")
    # vllt noch quantisiert als GGUF für OLLAMA speichern?

    print(f"{TColors.OKGREEN}Finetuning finished.{TColors.ENDC}")
    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"{TColors.HEADER}Computation Time: {duration}{TColors.ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm-confidentiality")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama3-8b-fine",
                        help="specifies the opponent LLM type")
    parser.add_argument("--iterations", "-i", type=int, default=1000,
                        help="specifies the number of iterations to finetune the LLM")
    parser.add_argument("--attacks", "-a", type=str, default=["all"],
                        help="specifies the attack types", nargs="+")
    parser.add_argument("--name_suffix", "-n", help="adds a name suffix for the finetuned model",
                        default="", type=str)
    args = parser.parse_args()
    main(**vars(args))
