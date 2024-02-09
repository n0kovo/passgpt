import sys
sys.path.append("../code")

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig, RobertaTokenizer
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from pathlib import Path

import numpy as np
import random

import time
from datetime import timedelta
import yaml
import shutil
from datasets import disable_caching

from utils import *  # Assuming this includes necessary utilities like PasswordDataCollator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", help="subsample the dataset to this number of entries", type=int, default=-1)
    args = parser.parse_args()

    with open("configs/passgpt-16chars.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = dotdict(config["config_args"])
    training_args = dotdict(config["training_args"])
    training_args["seed"] = args.seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert not os.path.exists(training_args.output_dir), "Output path already exists."
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    TOKENIZER_MAX_LEN = args.maxchars + 2

    print("===> Loading tokenizer")

    tokenizer = RobertaTokenizer.from_pretrained(
        '/drive/MyDrive/passgpt_llama2/tokenize_old/byte_bpe_tokenizer_98/',
)

    # Define dataloader
    print("===> Loading data")

    def preprocess_function(entries):
        """
        This function tokenizes a list of passwords. It appends the end of password token to each of them before processing.
        """
        to_tokenize = ['<s>' + p[:args.maxchars] +'</s>' for p in entries['text']]
        return tokenizer(to_tokenize,
                         truncation=True,
                         padding="max_length",
                         max_length=TOKENIZER_MAX_LEN,
                         add_special_tokens=False,
                         return_special_tokens_mask=False)


    data_files = {'train': ["/drive/MyDrive/passgpt_llama2/hashmob.net_2024-02-05.user.found"]}
    dataset = load_dataset('text', data_files=data_files)
    print("Dataset loaded with {} entries".format(len(dataset["train"])))

    if args.subsample > 0:
        print("Subsampling dataset to {} random entries".format(args.subsample))
        dataset['train'] = dataset['train'].select([i for i in range(args.subsample)])


    # Process data
    print("===> Processing data")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_datasets = tokenized_datasets.shuffle(seed=args.seed)

    # Format data
    tokenized_datasets.set_format(type="torch")


    print("===> Initializing model")
    model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")

    print("Model initialized with {} parameters".format(sum(t.numel() for t in model.parameters())))

    # Training setup
    from transformers import Trainer, TrainingArguments

    print("===> Preparing training")
    # Define the data collator. In charge of hiding tokens to be predicted.
    data_collator = PasswordDataCollator(
        tokenizer=tokenizer, mlm=False
    )

    train_args = TrainingArguments(
            **training_args
        )


    trainer = Trainer(
        model=model,
        args=training_args,  # Ensure this is adapted for LLaMA
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    # Launch training
    print("===> Launching training")
    start = time.time()
    trainer.train()
    end = time.time()

    print("===> Training completed after {}. Storing last version.".format(str(timedelta(seconds=end-start))))
    model.save_pretrained(os.path.join("/drive/MyDrive/passgpt_llama2/output", "last"))

    # Comment out next lines if you want to keep several checkpoints.
    print("===> Deleting previous checkpoints")
    checkpoints = [i for i in os.listdir("/drive/MyDrive/passgpt_llama2/output") if i.startswith("checkpoint")]
    for c in checkpoints:
        shutil.rmtree(os.path.join("/drive/MyDrive/passgpt_llama2/output", c))

    print("===> Training finished succesfully :)")
