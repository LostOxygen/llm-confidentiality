"""test script to finetune llama2 models on the yelp dataset"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import datetime
import socket
from typing import (
    Tuple,
)

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
import evaluate

from colors import TColors

os.environ["TRANSFORMERS_CACHE"] = "/data/"
LLM_TYPE = "meta-llama/Llama-2-7b-hf"
OUTPUT_NAME= "llama2-7b-specific"


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_function(data: str) -> str:
    return tokenizer(
        data["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


# setting devies and variables correctly
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
print(f"## {TColors.OKBLUE}{TColors.BOLD}LLM{TColors.ENDC}: {LLM_TYPE}")
print(f"## {TColors.OKBLUE}{TColors.BOLD}Output Name{TColors.ENDC}: {OUTPUT_NAME}")
print("#"*os.get_terminal_size().columns+"\n")

tokenizer = AutoTokenizer.from_pretrained(
        LLM_TYPE,
        use_fast=False,
    )
tokenizer.pad_token = tokenizer.unk_token

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
        LLM_TYPE,
        device_map="auto",
        quantization_config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )

# disable caching for finetuning
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.pad_token_id = tokenizer.pad_token_id
metric = evaluate.load("accuracy")

dataset = load_dataset("imdb", split="train")
#tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
)

peft_args = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    #train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(10)),
    #eval_dataset=tokenized_dataset["test"].shuffle(seed=42).select(range(10)),
    #compute_metrics=compute_metrics,
    max_seq_length=512,
    dataset_text_field="text",
    packing=True,
    #peft_config=peft_args,
)

trainer.train()
trainer.model.save_pretrained(
        os.path.join("", OUTPUT_NAME),
        safe_serialization=True,
        save_adapter=True,
        save_config=True
    )
trainer.tokenizer.save_pretrained(os.path.join("", OUTPUT_NAME))

