import wandb
import os
import torch
import random
import numpy as np
import torch
from typing import Any, Optional, Union
from time import gmtime, strftime
from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets import Dataset
from sklearn.metrics import log_loss
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    TrainerCallback
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from peft import (
    PeftModel,
    get_peft_model,
    LoraConfig,
    TaskType
)
from utils_prime import load_split_data


class AWP:
    def __init__(self, model, adv_param="weight", adv_lr=0.1, adv_eps=0.0001):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


TRAINING_ARGS_NAME = "traning_args.bin"


class CustomTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        awp_lr=0,
        awp_eps=0,
        awp_start_epoch=0
    ):

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        self.awp_lr = awp_lr
        self.awp_eps = awp_eps
        self.awp_start_epoch = awp_start_epoch

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        model_to_save = self.model
        state_dict = {
            k: v.to("cpu") for k, v in model_to_save.named_parameters() if v.requires_grad
        }
        # Using Hugging Face"s save_pretrained instead of PyTorch"s torch.save
        model_to_save.save_pretrained(
            output_dir,
            state_dict=state_dict,
            save_function=torch.save,
            safe_serialization=False
        )

        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        print(self.args)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model"s documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        o_inputs = inputs.copy()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        ########################
        # AWP
        if self.awp_lr != 0 and self.state.epoch >= self.awp_start_epoch:
            print("Start AWP!!!")
            self.awp = AWP(model, adv_lr=self.awp_lr, adv_eps=self.awp_eps)
            self.awp._save()
            self.awp._attack_step()
            with self.compute_loss_context_manager():
                awp_loss = self.compute_loss(self.awp.model, o_inputs)

            if self.args.n_gpu > 1:
                awp_loss = awp_loss.mean()
            elif self.use_apex:
                with amp.scale_loss(awp_loss, self.optimizer) as awp_scaled_loss:
                    awp_scaled_loss.backward()
            else:
                self.accelerator.backward(awp_loss)
            self.awp._restore()
        ########################

        return loss.detach() / self.args.gradient_accumulation_steps


def seed_everything(seed=None):
    """
    固定seed
    :param seed: int, 随机种子
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


seed_everything(42)


class InstructionDataSet(Dataset):
    def __init__(self, model_name, data, tokenizer, max_source_length, max_target_length, all_in_one):
        super(InstructionDataSet, self).__init__()
        self.model_name = model_name
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.all_in_one = all_in_one

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data.loc[index]

        if "gemma" in self.model_name:
            templete_part1 = "<start_of_turn>user\nHere are two question-answering dialogues. Compare two model performance on answering question, determine which is better.\n\n"
        elif "Phi" in self.model_name:
            templete_part1 = "<|system|>\nYou are a helpful assistant good at judging conversations.<|end|>\n<|user|>\nHere are two question-answering dialogues. Compare two model performance on answering question, determine which is better.\n"

        templete_part1_input_ids = self.tokenizer(
            text=templete_part1,
            add_special_tokens=True,
            padding=False
        )["input_ids"]

        if "gemma" in self.model_name:
            templete_part2 = "\n###options\nA. Model A\nB. Model B\nC. Tie\n<end_of_turn>\n"
        elif "Phi" in self.model_name:
            templete_part2 = "\n###options\nA. Model A\nB. Model B\nC. Tie\n<|end|>\n"

        templete_part2_input_ids = self.tokenizer(
            text=templete_part2,
            add_special_tokens=True,
            padding=False
        )["input_ids"][1:]

        if "gemma" in self.model_name:
            templete_part3 = "<start_of_turn>model\n"
        elif "Phi" in self.model_name:
            templete_part3 = "<|assistant|>\n"
            
        templete_part3_input_ids = self.tokenizer(
            text=templete_part3,
            add_special_tokens=True,
            padding=False
        )["input_ids"][1:]

        if self.all_in_one:
            prompt_response = now_data["prompt_response"]
            prompt_response_ids = self.tokenizer(
                text=prompt_response,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_source_length,
                padding=False
            )["input_ids"][1:]
        else:
            r_a = now_data["instruction_a"]
            r_b = now_data["instruction_b"]
            model_a_input_ids = self.tokenizer(
                text=r_a,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_source_length // 2,
                padding=False
            )["input_ids"]
            model_b_input_ids = self.tokenizer(
                text=r_b,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_source_length // 2,
                padding=False
            )["input_ids"]
            prompt_response_ids = model_a_input_ids + model_b_input_ids

        label = now_data["label"]
        label_ids = self.tokenizer.encode(text=label, add_special_tokens=False)
        input_ids = templete_part1_input_ids + prompt_response_ids + templete_part2_input_ids + templete_part3_input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids) - 2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }


@dataclass
class DataCollatorForInstruction:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = ((max_label_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of)

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"])
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        return features


def compute_metrics(p):
    logits = p.predictions
    logits = logits[:, 0, [A_TOKEN_IDS, B_TOKEN_IDS, C_TOKEN_IDS]]
    logits = torch.softmax(torch.tensor(logits).reshape(-1, 3), dim=-1)
    labels = torch.tensor(p.label_ids)

    bs, seq_len = labels.shape
    mask = labels != -100
    _, indices = torch.max(mask, dim=1)

    row_indices = torch.arange(bs).unsqueeze(1)
    col_indices = (indices.unsqueeze(1) + torch.arange(1)).clamp(max=seq_len-1)

    labels = labels[row_indices, col_indices]

    token2num = {A_TOKEN_IDS[0]: 0, B_TOKEN_IDS[0]: 1, C_TOKEN_IDS[0]: 2}
    labels = labels.reshape(-1).numpy()
    labels = np.array([token2num.get(val, val) for val in labels])
    prediction = logits.tolist()
    return {"log_loss": log_loss(labels, prediction, labels=[0, 1, 2])}


class SaveModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint,
                "sft_lora_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


def preprocess_logits_for_metrics(logits, labels):
    logits = logits.cpu()
    labels = labels.cpu()
    bs, seq_len, vocab_size = logits.shape
    mask = labels != -100
    _, indices = torch.max(mask, dim=1)

    row_indices = torch.arange(bs).unsqueeze(1)
    col_indices = (indices.unsqueeze(1) + torch.arange(2)).clamp(max=seq_len-1)

    logits = logits[row_indices, col_indices, :]
    return logits


def train(args):
    s = strftime("%a_%d_%b_%H_%M", gmtime())

    wandb.login(key="c465dd55c08ec111e077cf0454ba111b3a764a78")
    run = wandb.init(
        project=f"{args.MODEL.split('/')[-1]}_instruction",
        job_type="training",
        anonymous="allow"
    )

    df_train, df_valid = load_split_data(
        args.data_path,
        args.prompt_type,
        args.MAX_INPUT,
        args.if_train,
        args.split
    )

    if args.test_mode:
        df_valid = df_valid.loc[:20, :].reset_index(drop=True)

    if args.split == False:
        _, df_valid = load_split_data(
            "dataset/train.csv",
            args.prompt_type,
            args.MAX_INPUT,
            True,
            True
        )

    config = AutoConfig.from_pretrained(args.MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.MODEL,
        trust_remote_code=True,
        truncation_side="left"
    )

    print(f"arg is {args}")

    train_dataset_path = "./dataset_cache/" + args.train_data.split("/")[-1].split(".")[0] + "_" + args.MODEL.replace("/", "-") + "_" + args.token_type
    valid_dataset_path = "./dataset_cache/" + args.valid_data.split("/")[-1].split(".")[0] + "_" + args.MODEL.replace("/", "-") + "_" + args.token_type

    if not os.path.exists(train_dataset_path):
        os.makedirs(train_dataset_path)
    if not os.path.exists(valid_dataset_path):
        os.makedirs(valid_dataset_path)

    train_cache_path = os.path.join(train_dataset_path, "dataset.bin")
    valid_cache_path = os.path.join(valid_dataset_path, "dataset.bin")

    if args.use_cache and os.path.exists(train_cache_path):
        tokenized_dataset = torch.load(train_cache_path)
    else:
        tokenized_dataset = InstructionDataSet(
            args.MODEL,
            df_train,
            tokenizer,
            args.MAX_INPUT,
            1,
            args.all_in_one
        )
        torch.save(tokenized_dataset, train_cache_path)

    if args.use_cache and os.path.exists(valid_cache_path):
        tokenized_dataset_valid = torch.load(valid_cache_path)
    else:
        tokenized_dataset_valid = InstructionDataSet(
            args.MODEL,
            df_valid,
            tokenizer,
            args.MAX_INPUT,
            1,
            args.all_in_one
        )
        torch.save(tokenized_dataset_valid, valid_cache_path)

    global A_TOKEN_IDS
    A_TOKEN_IDS = tokenizer(
        "A",
        add_special_tokens=True,
        truncation=True,
        max_length=1024
    )["input_ids"][1:]

    global B_TOKEN_IDS
    B_TOKEN_IDS = tokenizer(
        "B",
        add_special_tokens=True,
        truncation=True,
        max_length=1024
    )["input_ids"][1:]

    global C_TOKEN_IDS
    C_TOKEN_IDS = tokenizer(
        "C",
        add_special_tokens=True,
        truncation=True,
        max_length=1024
    )["input_ids"][1:]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.MODEL,
        config=config,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    checkpoint = None
    if len(args.resume_from_checkpoint) != 0:
        checkpoint = args.resume_from_checkpoint
        print(f"Using Checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
        print(model)

    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            # bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        model = get_peft_model(model, peft_config)

    print(model.print_trainable_parameters())

    model.enable_gradient_checkpointing()

    data_collator = DataCollatorForInstruction(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        output_dir=f"output/{wandb.run.name}",
        report_to="wandb",
        overwrite_output_dir=True,
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        metric_for_best_model="log_loss",
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        save_total_limit=10,
        label_smoothing_factor=args.label_smoothing_factor
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.num_train_epochs *
        int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
            training_args.gradient_accumulation_steps) * args.warm_up_ratio,
        num_training_steps=training_args.num_train_epochs *
        int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
            training_args.gradient_accumulation_steps),
        num_cycles=1.5
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        awp_lr=args.awp_lr,
        awp_eps=args.awp_eps,
        awp_start_epoch=args.awp_start_epoch,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    wandb.finish()
