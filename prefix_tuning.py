"""main hook to start the prefix tuning"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# code is heavily based on https://github.com/XiangLi1999/PrefixTuning,
# https://github.com/eth-sri/sven/tree/master and
# https://huggingface.co/docs/peft/task_guides/seq2seq-prefix-tuning

import os
import sys
import time
import datetime
import socket
import argparse
from typing import Final, List, Callable

import pkbar
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup
)
from peft import (
    get_peft_model,
    PrefixTuningConfig,
    TaskType
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from framework.prompts import SECRET_KEY
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
from framework.dataset import PromptDataset, ResponseDataset
from framework.llm import LLM

# number of attack samples per attack type and main iteration
NUM_ATTACK_SAMPLES: Final[int] = 1000
DATA_PATH: Final[str] = "./datasets/system_prompts_train.json"
OUTPUT_DIR: Final[str] = "./finetuned_models/"
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.isdir("/data/"):
    os.mkdir("/data/")
os.environ["TRANSFORMERS_CACHE"] = "/data/"

# hacky global variables for the tokenizer
glob_tokenizer: AutoTokenizer = None
glob_max_length: int = 4096


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

    for attack in attacks:
        # set the attack function
        match attack:
            case "payload_splitting": attack_funcs.append(payload_splitting)
            case "obfuscation": attack_funcs.append(obfuscation)
            case "manipulation": attack_funcs.append(manipulation)
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


def preprocess_function(data) -> dict:
    """
    Helper function to preprocess the data for the LLM by mapping
    the prompts to their encodings (Tokens).
    """
    inputs = data["prompts"]
    labels = data["labels"]
    glob_tokenizer.pad_token = glob_tokenizer.eos_token
    model_inputs = glob_tokenizer(
            inputs,
            max_length=glob_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    label_inputs = glob_tokenizer(
            labels,
            max_length=glob_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    model_inputs["labels"] = label_inputs["input_ids"]
    return model_inputs


def get_data(attacks: List[Callable] = None, batch_size: int = 8) -> DataLoader:
    """
    Creating a dictionary dataset from the system prompts.

    Parameters:
        attacks: List[Callable] - the list of attacks to harden the LLM against
        batch_size: int - the batch size for the dataloader

    Returns:
        train_data: DataLoader - the DataLoader containing the tokenized prompt dataset
    """
    assert os.path.isfile(DATA_PATH), f"{TColors.FAIL}Couldn't find dataset.{TColors.ENDC}"
    prompt_dataset = PromptDataset(is_train=True)
    response_dataset = ResponseDataset()
    prompt_list: List[str] = []
    label_list: List[str] = []

    # create prompt injection attack dataset
    for attack in attacks:
        for _ in range(NUM_ATTACK_SAMPLES):
            random_prompt = prompt_dataset.get_random_prompt()
            random_response = response_dataset.get_random_response()
            sanitized_response = random_response.replace(SECRET_KEY, "#"*len(SECRET_KEY))
            attack_str = attack()

            prompt = f"""<s>[INST] <<SYS>>
                {random_prompt}

                <</SYS>>
                {attack_str}

                [/INSTR]
                {random_response}
                </s>
            """
            prompt_list.append(prompt)

            label = f"""<s>[INST] <<SYS>>
                {random_prompt}

                <</SYS>>
                {attack_str}

                [/INSTR]
                {sanitized_response}
                </s>
            """
            label_list.append(label)

    new_dataset = Dataset.from_dict({"prompts": prompt_list, "labels": label_list})

    # convert the prompts into tokens
    processed_dataset = new_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=new_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on the prompts",
    )

    # create the dataloader
    train_dataloader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
        pin_memory=True
    )

    return train_dataloader


def main(
        llm_type: str,
        epochs: int,
        attacks: List[str],
        name_suffix: str,
        learning_rate: float,
        batch_size: int,
        max_length: int,
        prefix_length: int
    ) -> None:
    """
    Main function to start the LLM prefix tuning
    
    Parameters:
        llm_type: str - specifies the opponent LLM type
        epochs: int - specifies the number of iterations to finetune the LLM
        attacks: List[str] - specifies the attack functions to harden the LLM against
        name_suffix: str - adds a name suffix for the final model
        max_length: int - specifies the maximum length of the input sequence
        learning_rate: float - specifies the learning rate for the optimizer
        batch_size: int - specifies the batch size for the dataloader
        prefix_length: int - specifies the number of virtual tokens to train as a prefix

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

    # setting devies and variables correctly
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda:0"
    accelerator = Accelerator()

    # setting the suffixes
    suffix: str = "prefix"
    name_suffix: str = "-" + name_suffix if name_suffix != "" else ""
    attack_suffix: str = "-" + "".join(attacks)

    # combine the finale output save name
    save_name: str = llm_type + "-" + suffix + attack_suffix + name_suffix

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
    print(f"## {TColors.OKBLUE}{TColors.BOLD}LLM{TColors.ENDC}: {llm_type}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Attacks: {TColors.ENDC}: {attacks}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Output Name{TColors.ENDC}: {save_name}")

    # print the prefix tuning parameters
    print(f"## {TColors.HEADER}{TColors.BOLD}{TColors.UNDERLINE}Prefix-Tuning Parameters " \
          f"{TColors.ENDC}" + "#"*int(os.get_terminal_size().columns-28))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}epochs{TColors.ENDC}: {epochs}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}max_length{TColors.ENDC}: {max_length}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}batch_size{TColors.ENDC}: {batch_size}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}learning_rate{TColors.ENDC}: {learning_rate}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}prefix_length{TColors.ENDC}: {prefix_length}")
    print("#"*os.get_terminal_size().columns+"\n")

    # load the llm and config and stuff
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=prefix_length
    )

    llm = LLM(llm_type=llm_type)

    model = get_peft_model(llm.model, peft_config)
    model.print_trainable_parameters()

    # set some global variables for the dataset mappings
    global glob_tokenizer
    glob_tokenizer = llm.tokenizer
    global glob_max_length
    glob_max_length = max_length

    # create list of attacks to harden against if robust finetuning is enabled
    attack_funcs = get_attack_list(attacks)

    # create the dataloaders
    train_data = get_data(attacks=attack_funcs, batch_size=batch_size)

    # setting up the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=(len(train_data) * epochs),
    )

    # prepare to accelerate
    model, optimizer, train_data = accelerator.prepare(
        model, optimizer, train_data
    )

    kbar = pkbar.Kbar(target=epochs, width=40, always_stateful=True)
    # create the training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for _, batch in enumerate(train_data):
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_data)
        train_ppl = torch.exp(train_epoch_loss)
        train_epoch_loss = train_epoch_loss.item()
        train_ppl = train_ppl.item()
        kbar.update(epoch+1, values=[("train_epoch_loss", train_epoch_loss),
                                     ("train_ppl", train_ppl)])

    # save the model
    model.save_pretrained(os.path.join(OUTPUT_DIR, save_name), safe_serialization=True)
    llm.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, save_name))

    print(f"\n{TColors.OKGREEN}Prefix-Tuning finished.{TColors.ENDC}")
    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"{TColors.HEADER}Computation Time: {duration}{TColors.ENDC}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefix Tuning")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama2-7b",
                        help="specifies the opponent LLM type")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="specifies the number of iterations to finetune the LLM")
    parser.add_argument("--attacks", "-a", type=str, default=["payload_splitting"],
                        help="specifies the attack types", nargs="+")
    parser.add_argument("--name_suffix", "-n", help="adds a name suffix for the finetuned model",
                        default="", type=str)
    parser.add_argument("--batch_size", "-bs", help="specifies the training batch size",
                        default=1, type=int)
    parser.add_argument("--learning_rate", "-lr", help="specifies the training learning rate",
                        default=5e-5, type=float)
    parser.add_argument("--max_length", "-ml", help="specifies the max. sequence length",
                        default=1024, type=int)
    parser.add_argument("--prefix_length", "-pl", default=10, type=int,
                        help="specifies the prefix length (virtual tokens)")
    args = parser.parse_args()
    main(**vars(args))
