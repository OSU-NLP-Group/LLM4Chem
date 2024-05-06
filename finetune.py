import os
import sys
from typing import List
import numpy as np
import random
import fire
import torch
import transformers
from datasets import load_dataset
import datetime

from utils.chat_generation import generate_chat


torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))


from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict
)

from transformers import AutoTokenizer, AutoModelForCausalLM

from trainer import CustomTrainer, CustomDataCollator

from utils.general_prompter import GeneralPrompter, get_chat_content
from utils.core_tagger import CoreTagger


def set_random_seeds(seed: int = 13):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seeds()


def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    output_dir: str = "checkpoint",
    # training hyperparams
    batch_size: int = 512,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    use_val_set: bool = True,
    optim="adamw_bnb_8bit",
    lr_scheduler: str = "cosine",
    warmup_steps: int = 1000,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    modules_to_save: List[str] = [],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    logging_steps: int = 10,
    save_steps: int = 200,
    save_total_limit=None,
    eval_steps: int = 200,
    use_int8: bool = False,
    precision='bf16',
    train_split='train',
    dev_split='validation',
    tasks: List[str] = None,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            # f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"use_val_set: {use_val_set}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"precision: {precision}\n"
            f"use_int8: {use_int8}\n"
        )
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if precision == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError("Please use bf16. Others are not tested.")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=use_int8,
        torch_dtype=dtype,
        device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    tokenizer.sep_token = '<unk>'
    tokenizer.cls_token = '<unk>'
    tokenizer.mask_token = '<unk>'
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        # print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")
        assert (bos, eos, pad) == (1, 2, None), (bos, eos, pad)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"

    prefix_chat = None
    prompter = GeneralPrompter(get_chat_content, '[/INST]')
    core_tagger = CoreTagger(tokenizer, core_tags_as_special_tokens=False, include_tags=True)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point, add_core_mask=True):
        input_text = data_point['input']
        output_text = data_point['output']
        chat = generate_chat(input_text, output_text, prefix_chat=prefix_chat)
        full_prompt = prompter.generate_prompt(chat)
        tokenized_full_prompt = tokenize(full_prompt)

        if add_core_mask or not train_on_inputs:
            user_prompt = prompter.generate_prompt(generate_chat(input_text, output_text=None))
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if not train_on_inputs:
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # TODO: could be sped up, probably

            if add_core_mask:
                core_mask = core_tagger.generate_mask(tokenized_full_prompt['input_ids'], user_prompt_len, data_point)
                tokenized_full_prompt['core_mask'] = core_mask

        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            # resume_from_checkpoint = (
            #     False  # So the trainer won't try loading its state
            # )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
            
    model.print_trainable_parameters()

    if tasks is not None and len(tasks) == 0:
        tasks = None

    train_data = load_dataset(data_path, split=train_split, tasks=tasks)
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt)

    if use_val_set:
        val_data = load_dataset(data_path, split=dev_split, tasks=tasks)
        val_data = val_data.shuffle().map(generate_and_tokenize_prompt)
    else:
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True if 'fp16' == precision else False,
            bf16=True if 'bf16' == precision else False,
            logging_steps=logging_steps,
            optim=optim,
            evaluation_strategy="steps" if val_data is not None else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_data is not None else None,
            save_steps=save_steps,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if val_data is not None else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=CustomDataCollator(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir, save_embedding_layers=True)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
