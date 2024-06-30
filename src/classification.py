import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import argparse
import warnings
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset
from sklearn.metrics import log_loss
from scipy.special import softmax
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from utils import load_config, classification_data_preprocessing
warnings.filterwarnings("ignore")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    logits = p.predictions
    print(logits)
    labels = p.label_ids
    print(labels)
    probabilities = softmax(logits, axis=-1)
    print(probabilities)

    return {"log_loss": log_loss(labels, probabilities)}


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
        token=args.huggingface_api
    )
    if "Llama" in args.model_name:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.add_special_tokens({"sep_token": "[SEP]"})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=args.huggingface_api
    )
    print(model)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    print("New tokenizer length:", len(tokenizer))

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    def tokenize(sample):
        return tokenizer(sample["input"], max_length=args.max_length, truncation=True)

    train, val = classification_data_preprocessing(args.train_file_path)
    train["input"] = train["full_chat_a"] + "[SEP]" + train["full_chat_b"]
    val["input"] = val["full_chat_a"] + "[SEP]" + val["full_chat_b"]

    ds_train = Dataset.from_pandas(train)
    ds_val = Dataset.from_pandas(val)

    ds_train = ds_train.map(tokenize).remove_columns(
        ["id", "prompt", "response_a", "response_b", "full_chat_a", "full_chat_b"]
    )
    ds_val = ds_val.map(tokenize).remove_columns(
        ["id", "prompt", "response_a", "response_b", "full_chat_a", "full_chat_b"]
    )

    class DataCollator:
        def __call__(self, features):
            model_inputs = [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                    "labels": feature["labels"]
                } for feature in features
            ]
            batch = tokenizer.pad(
                model_inputs,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="pt",
                pad_to_multiple_of=8
            )
            return batch

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        output_dir="output",
        report_to="none",
        overwrite_output_dir=True,
        greater_is_better=False,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.num_train_epochs * int(len(ds_train) * 1.0 / training_args.per_device_train_batch_size / training_args.gradient_accumulation_steps)
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollator(),
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Training")
    parser.add_argument(
        "--config",
        default="classification.yaml",
        type=str,
        help="path to .yaml file",
        required=False
    )
    args = parser.parse_args()

    config = load_config(args.config)

    for key, value in config.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))

    args = parser.parse_args()

    train(args)
