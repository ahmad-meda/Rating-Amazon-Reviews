import os
import shutil
import argparse
import yaml
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from huggingface_hub import login
from sklearn.metrics import accuracy_score

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    pipeline,
    Trainer, TrainingArguments, 
    AutoTokenizer, AutoModelForSequenceClassification, 
)


parser = argparse.ArgumentParser(description="Training script with YAML config.")
parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML).")
args =  parser.parse_args()

# Load configuration from YAML
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Login to Hugging Face
hf_token = "INSER_YOUR_HF_TOKEN_HERE"
login(hf_token)

# Construct hub_model_id from config
if not config["hub_model_id"]:
    config["hub_model_id"] = f"f21aa/{config['model_id'].split('/')[-1].lower()}-{'-'.join(map(str, config['dataset_splits']))}-v{config['exp_id']}"
if config["debug"]:
    config["hub_model_id"] = "f21aa/testing-lambdalabs"

# Load the dataset
train_splits = []
for split in config["dataset_splits"]:
    train_splits.append(load_dataset(config["dataset_name"], split=split))
train_dataset = concatenate_datasets(train_splits).shuffle(seed=1)

# If in debug mode, slice the dataset
if config["debug"]:
    train_dataset = train_dataset.select(range(900))

# Convert labels to 0-index
train_dataset = train_dataset.map(lambda x: {"label": x["label"] - 1})

# Split into train/test
dataset = train_dataset.train_test_split(
    test_size=config["test_size"], 
    seed=1, 
    shuffle=True
)

tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
model = AutoModelForSequenceClassification.from_pretrained(
    config["model_id"],
    num_labels=5
).to("cuda:0")

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=config["max_length"], 
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = accuracy_score(labels, predictions)
    return {"accuracy": float(score) if score == 1 else score}

training_args = TrainingArguments(
    output_dir=config["hub_model_id"].split("/")[-1],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    
    learning_rate=config["lr"],
    num_train_epochs=config["num_epochs"],
    bf16=config["bf16"],
    fp16=not config["bf16"],
    optim="adamw_torch_fused",  # improved optimizer

    # logging & evaluation
    eval_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    eval_steps=config["num_steps"],
    save_steps=config["num_steps"],
    logging_steps=int(config["num_steps"] // 5),

    save_total_limit=25,
    report_to="none",

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    # push to hub
    push_to_hub=True,
    hub_model_id=config["hub_model_id"],
    hub_strategy="every_save",
    hub_private_repo=True,
    hub_token=hf_token,
)

trainer = Trainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

shutil.copy(
    "config.yaml", 
    os.path.join(training_args.output_dir, f"training_config.yaml")
)

trainer.train()

test_ds = load_dataset('f21aa/test', split='train')
tokenized_test_ds = test_ds.map(tokenize, batched=True)

predictions, _, _ = trainer.predict(tokenized_test_ds)

test_df = test_ds.to_pandas()
test_df['Score'] = np.argmax(predictions, axis=1)+1

test_df[['Id', 'Score']].to_csv(f"{config['hub_model_id'].split('/')[-1]}.csv", index=False)


# Persist important files
shutil.copy(
    f"{config['hub_model_id'].split('/')[-1]}.csv", 
    os.path.join(training_args.output_dir, f"{config['hub_model_id'].split('/')[-1]}.csv")
)

trainer.push_to_hub(commit_message="Persisting preds and config file")