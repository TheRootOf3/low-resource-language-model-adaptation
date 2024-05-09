"""
Language Model training script.

Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py
by Andrzej Szablewski, 2024 as a part of the UCL Final Year Project 
"Language Model Adaptation for Low-Resource African Languages". 

--- Original header: ---

Copyright 2021 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import math
import os
import time

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    default_data_collator,
    get_scheduler,
)

from training import parse_args
from training import get_dict_from_args_str, get_config_model, get_tokenizer

from peft import LoraConfig, get_peft_model

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load tokenizer
    tokenizer = get_tokenizer(
        tokenizer_name_or_path=(
            args.tokenizer_name
            if args.tokenizer_name is not None
            else args.model_name_or_path
        ),
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # Load model config and model
    config, model = get_config_model(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # If use LoRA.
    if args.use_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["embed_tokens", "embed_positions", "lm_head"],
        )
        model = get_peft_model(model, peft_config)

        if accelerator.is_main_process:
            print("Using LoRA.")
            model.print_trainable_parameters()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print("Resizing model embeddings!")
        model.resize_token_embeddings(len(tokenizer))

    # Load the model
    lm_datasets = load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
    )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Set the block size
    block_size = min(1024, config.max_position_embeddings)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
    )

    # Print updateable params.
    if accelerator.is_main_process:
        for i, (n1, p1) in enumerate(list(model.model.named_parameters())):
            if p1.requires_grad:
                print(i, n1)

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Logging interval
    logging_interval = args.logging_interval
    if logging_interval is not None and logging_interval.isdigit():
        logging_interval = int(logging_interval)

    # Evaluation interval
    eval_interval = args.eval_interval
    if eval_interval is not None and eval_interval.isdigit():
        eval_interval = int(eval_interval)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value

        # With wandb tracking
        wandb_args_dict = get_dict_from_args_str(args.wandb_args)
        wandb_project_name = wandb_args_dict["project"]
        del wandb_args_dict["project"]
        accelerator.init_trackers(
            wandb_project_name,
            experiment_config,
            {"wandb": wandb_args_dict},
        )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    t0 = time.time()
    samples_seen = 0

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(
            accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size))
        )

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"epoch {0}: perplexity: {perplexity} eval_loss: {eval_loss}")

    if args.with_tracking:
        accelerator.log(
            {
                "Perplexity/eval": perplexity,
                "Loss/eval": eval_loss,
                "epoch": 0,
                "step": completed_steps,
            },
            step=completed_steps,
        )
    model.train()

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            running_loss_per_epoch = 0
            loss_per_batch = 0
            steps_per_epoch = 0

        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    steps_per_epoch += 1
                    running_loss_per_epoch -= running_loss_per_epoch / steps_per_epoch
                    running_loss_per_epoch += loss.detach().float() / steps_per_epoch
                    loss_per_batch = loss.detach().float()
                    samples_seen += total_batch_size

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"

                        # save state
                        if args.output_dir is not None:
                            output_dir = os.path.join(
                                args.output_dir, output_dir, "state"
                            )
                        accelerator.save_state(output_dir)

                        # save pre-trained model
                        if args.output_dir is not None:
                            output_dir = os.path.join(
                                args.output_dir, output_dir, "model"
                            )
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            output_dir,
                            is_main_process=accelerator.is_main_process,
                            safe_serialization=False,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(model),
                        )
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(output_dir)

                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if completed_steps % logging_interval == 0 and completed_steps > 0:
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "Loss/train": running_loss_per_epoch.item(),
                                "Loss/train_per_batch": loss_per_batch,
                                "iter_time": dt,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "step": completed_steps,
                                "samples_seen": samples_seen,
                                "approx_tokens_seen": samples_seen * block_size,
                            },
                            step=completed_steps,
                        )

                if completed_steps % eval_interval == 0:
                    model.eval()
                    losses = []
                    for step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)

                        loss = outputs.loss
                        losses.append(
                            accelerator.gather_for_metrics(
                                loss.repeat(args.per_device_eval_batch_size)
                            )
                        )

                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    logger.info(
                        f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
                    )

                    if args.with_tracking:
                        accelerator.log(
                            {
                                "Perplexity/eval": perplexity,
                                "Loss/eval": eval_loss,
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                    model.train()

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "Perplexity/eval": perplexity,
                    "Loss/eval": eval_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if args.use_lora:
            unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            safe_serialization=False,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(unwrapped_model),
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
