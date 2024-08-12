import numpy as np
import os
import torch
import torch.nn as nn
import argparse
import warnings
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import log_loss
from scipy.special import softmax
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from utils import load_config, classification_data_preprocessing
from bitsandbytes.optim import AdamW8bit
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
    labels = p.label_ids
    probabilities = softmax(logits, axis=-1)

    return {"log_loss": log_loss(labels, probabilities)}


def train(args):
    wandb.login(key="")
    run = wandb.init(
        project=f"{args.model_name.split('/')[-1]}_cls",
        job_type="training",
        anonymous="allow"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
        add_eos_token=True,
        token=args.huggingface_api
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=args.huggingface_api
    )
    print(model)

    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except:
        print("Set eos_token as pad_token !!!")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print("New Tokenizer Length:", len(tokenizer))

    if bnb_config.bnb_4bit_compute_dtype == torch.float16:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    elif bnb_config.bnb_4bit_compute_dtype == torch.bfloat16:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
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
    train["input"] = train["full_chat_a"] + "\n####\n" + train["full_chat_b"]
    val["input"] = val["full_chat_a"] + "\n####\n" + val["full_chat_b"]

    ds_train = Dataset.from_pandas(train)
    ds_val = Dataset.from_pandas(val)

    ds_train = ds_train.map(tokenize).remove_columns(
        ["id", "prompt", "response_a", "response_b", "full_chat_a", "full_chat_b"]
    )
    ds_val = ds_val.map(tokenize).remove_columns(
        ["id", "prompt", "response_a", "response_b", "full_chat_a", "full_chat_b"]
    )

    # class DataCollator:
    #     def __call__(self, features):
    #         model_inputs = [
    #             {
    #                 "input_ids": feature["input_ids"],
    #                 "attention_mask": feature["attention_mask"],
    #                 "labels": feature["labels"]
    #             } for feature in features
    #         ]
    #         batch = tokenizer.pad(
    #             model_inputs,
    #             padding="max_length",
    #             max_length=args.max_length,
    #             return_tensors="pt",
    #             pad_to_multiple_of=8
    #         )
    #         return batch

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        output_dir=f"{args.model_name.split('/')[-1]}_cls_output",
        report_to="wandb",
        overwrite_output_dir=True,
        greater_is_better=False,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        save_only_model=True,
        metric_for_best_model="log_loss",
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit
    )

    optimizer = AdamW8bit(
        model.parameters(),
        lr=args.learning_rate
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.num_train_epochs * int(len(ds_train) * args.warmup_ratio / training_args.per_device_train_batch_size / training_args.gradient_accumulation_steps),
        num_training_steps=training_args.num_train_epochs * int(len(ds_train) * 1.0 / training_args.per_device_train_batch_size / training_args.gradient_accumulation_steps),
        num_cycles=args.num_cycles
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
