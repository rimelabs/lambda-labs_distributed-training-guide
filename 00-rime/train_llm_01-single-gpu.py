import argparse
from itertools import chain
import json
import multiprocessing
import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
import wandb
import tqdm
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from packed_dataset import pad_and_collate

LOGGER = logging.getLogger(__name__)

def main():
    torch.set_float32_matmul_precision('high')

    parser = _get_parser()
    args = parser.parse_args()

    # Will be modifying this in future version to include rank information
    logging.basicConfig(
        format=f"[%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    # Helpful to log this information when running on multiple nodes to make sure all nodes have the same environment.
    LOGGER.info(os.environ)
    LOGGER.info(args)

    # This guide assumes CUDA device is available, and does all training in bf16
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Seed pytorch's RNG. See https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)

    # Note: Initializing an **untrained** model
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_cache=False)
    with device:
        from liger_kernel.transformers import apply_liger_kernel_to_llama

        # 1a. Adding this line automatically monkey-patches the model with the optimized Liger kernels
        apply_liger_kernel_to_llama()

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )

    number_add_tokens = 7 * 4096 + 10
    new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    LOGGER.info(f"{sum(p.numel() for p in model.parameters())} model parameters")

    train_data = datasets.Dataset.load_from_disk("/workspace/tmp/pretrain_ds_first-1M-packed-8192").with_format("torch")
    LOGGER.info(f"{len(train_data)} training samples")

    # Standard pytorch dataset iterator
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=pad_and_collate,
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    # NOTE: T_max and eta_min were arbitrarily chosen
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    exp_dir: Path = Path(args.save_dir) / args.experiment_name
    LOGGER.info(f"Experiment saving to {exp_dir}")

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if (exp_dir / "state.json").exists():
        # NOTE: weights_only is to protect against arbitrary code execution with pickle decoding.
        def _load_to_device(p):
            return torch.load(p, map_location=device, weights_only=True)

        model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
        lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)
        resumed = True
    LOGGER.info(f"Resumed={resumed} | {state}")

    LOGGER.info(f"Creating experiment root directory")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Initializing [wandb](https://wandb.ai/) - a very useful experiment tracking library.
    wandb.init(
        project="distributed-training-guide",
        dir=exp_dir,
        name=args.experiment_name,
        id=args.experiment_name,
        resume="must" if resumed else None,
        save_code=True,
        config={
            "args": vars(args),
            "training_data_size": len(train_data),
            "num_batches": len(dataloader),
        },
    )

    # will be using to understand breakdown of speed
    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        # NOTE: This is not standard. Normally you can just iterate directly over dataloader.
        #       We are doing this so we can explicitly measure the time it takes to generate a batch.
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            # Here we measure the time it takes to generate a batch and move it to the GPU
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            # For resuming, this has to come after getting the next batch, so we move through the dataset properly.
            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            with timers["forward"]:
                outputs = model(**batch)

            with timers["backward"]:
                # NOTE: set_to_none=True will de-allocate the gradients, saving us some memory.
                optimizer.zero_grad(set_to_none=True)
                outputs.loss.backward()

            with timers["update"]:
                optimizer.step()
                lr_scheduler.step()

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                tok_per_step = args.batch_size * 8_192
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": state["running_loss"] / args.log_freq,
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),
                    "num_batches_remaining": len(dataloader) - i_step,
                    **get_mem_stats(device),
                    "tok/s": 1000 * tok_per_step / ms_per_step,
                    "time/total": ms_per_step,
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }

                LOGGER.info(info)
                wandb.log(info, step=state["global_step"])

                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if state["global_step"] % args.ckpt_freq == 0:
                LOGGER.info("Saving checkpoint.")
                torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
                torch.save(model.state_dict(), exp_dir / "model.pt")
                torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                with open(exp_dir / "state.json", "w") as fp:
                    json.dump(state, fp)

        state["epoch_step"] = 0

def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", default=None, required=True)
    # parser.add_argument("-d", "--dataset-name", default=None, required=True)
    # parser.add_argument("-m", "--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=1, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=50, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    # parser.add_argument("-s", "--seq-length", default=1024, type=int)
    return parser


if __name__ == "__main__":
    main()
