import glob
import math
import os
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import random

from configs import TinyLMConfig
from lit_gpt import FusedCrossEntropyLoss
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import (chunked_cross_entropy,
                           get_default_supported_precision, num_parameters,
                           step_csv_logger)
from pytorch_lightning.loggers import WandbLogger


def get_config_from_yaml(
    config_path: Path = Path("configs/example.yaml"),
):
    config = TinyLMConfig.from_yaml(config_path)
    hparams = TinyLMConfig.to_dict(config)
    # hparams = {
    #     k: v
    #     for k, v in locals().items()
    #     if isinstance(v, (int, float, str)) and not k.startswith("_")
    # }
    logger = step_csv_logger("out", config.name, flush_logs_every_n_steps=config.log_iter_interval)
    wandb_logger = WandbLogger(project=config.project, entity=config.entity, name=config.name)

    return config, hparams, logger, wandb_logger


def setup(
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    config_path: Path = Path("configs/example.yaml"),
) -> None:
    config, hparams, logger, wandb_logger = get_config_from_yaml(config_path)
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if config.num_of_devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            if "mistral" in config.model.lower():
                strategy = FSDPStrategy(
                    auto_wrap_policy={Block},
                    # activation_checkpointing_policy=None,
                    activation_checkpointing_policy={Block},
                    state_dict_type="full",
                    sharding_strategy="FULL_SHARD",
                    limit_all_gathers=True,
                    cpu_offload=False,
                )
            elif "llama" in config.model.lower():
                strategy = FSDPStrategy(
                    auto_wrap_policy={Block},
                    activation_checkpointing_policy=None,
                    state_dict_type="full",
                    sharding_strategy="HYBRID_SHARD",
                    limit_all_gathers=True,
                    cpu_offload=False,
                )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        num_nodes=config.num_of_nodes,
        accelerator="gpu",
        devices=config.num_of_devices,
        strategy=strategy,
        precision=precision,
        loggers=[logger, wandb_logger],
    )
    fabric.print(hparams)
    # fabric.launch(main, train_data_dir, val_data_dir, resume)
    # main(fabric, config, hparams, resume)
    fabric.launch(main, config, hparams, resume)


def main(fabric, lm_config: TinyLMConfig, hparams, resume):
    monitor = Monitor(
        fabric,
        window_size=2,
        time_unit="seconds",
        log_iter_interval=lm_config.log_iter_interval,
    )

    if fabric.global_rank == 0:
        lm_config.out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(lm_config.model)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=lm_config.micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=lm_config.train_data_dir,
        val_data_dir=lm_config.val_data_dir,
        seed=1234,
        lm_config=lm_config,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(1234)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        if lm_config.tie_embedding:
            model.transformer.wte.weight = model.lm_head.weight
            model = torch.compile(model)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    if lm_config.initial_checkpoint_dir != "":
        fabric.load_raw(lm_config.initial_checkpoint_dir, model)

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lm_config.learning_rate,
        weight_decay=lm_config.weight_decay,
        betas=(lm_config.beta1, lm_config.beta2),
        foreach=False,
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

    if resume is True:
        resume = sorted(lm_config.out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
        if lm_config.reinit_optim:
            state["optimizer"] = optimizer

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume, lm_config)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, lm_config):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        t0 = time.perf_counter()
        val_loss = validate(fabric, model, val_dataloader, lm_config)  # sanity check
        t1 = time.perf_counter() - t0
        fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
        fabric.log_dict(
            {
                "metric/val_loss": val_loss.item(),
                "total_tokens": model.config.block_size
                * (state["iter_num"] + 1)
                * lm_config.micro_batch_size
                * fabric.world_size,
            },
            state["step_count"],
        )
        fabric.log_dict(
            {
                "metric/val_ppl": math.exp(val_loss.item()),
                "total_tokens": model.config.block_size
                * (state["iter_num"] + 1)
                * lm_config.micro_batch_size
                * fabric.world_size,
            },
            state["step_count"],
        )
        fabric.barrier()
    # if only do_eval and test ppl
    if lm_config.only_validate:
        # save the loss and ppl to a file named with the model name
        os.makedirs("../eval_analysis/ppl_results", exist_ok=True)
        save_name = str(resume).replace("iter-", "").replace("-ckpt.pth", "").replace("/", "_")
        # remove iter-005000-ckpt.pth to empty string
        file_name = re.sub(r"iter-\d{6}-ckpt.pth", "", str(resume).replace("/", "_"))
        with open(f"../eval_analysis/ppl_results/{file_name}_ppl.csv", "a") as f:
            f.write("model_name,ppl,loss\n")
            f.write(f"{save_name},{math.exp(val_loss.item())},{val_loss.item()}\n")
        return

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * lm_config.micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (lm_config.micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        # measured_flops = measure_flops(meta_model, x)
        # fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = lm_config.curr_iter

    print(curr_iter, initial_iter, lm_config.max_iters)

    loss_func = FusedCrossEntropyLoss()
    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= lm_config.max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], lm_config) if lm_config.decay_lr else lm_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()
        is_accumulating = (state["iter_num"] + 1) % lm_config.gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / lm_config.gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=lm_config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()
        state["iter_num"] += 1
        # input_id: B L
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (lm_config.max_iters - state['iter_num']) / 3600:.2f} hours. "
            # print days as well
            f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (lm_config.max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
        )

        monitor.on_train_batch_end(
            state["iter_num"] * lm_config.micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item(),
        )

        if (
            val_dataloader is not None
            and not is_accumulating
            and state["step_count"] % lm_config.eval_step_interval == 0
        ):
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, lm_config)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict(
                {
                    "metric/val_loss": val_loss.item(),
                    "total_tokens": model.config.block_size
                    * (state["iter_num"] + 1)
                    * lm_config.micro_batch_size
                    * fabric.world_size,
                },
                state["step_count"],
            )
            fabric.log_dict(
                {
                    "metric/val_ppl": math.exp(val_loss.item()),
                    "total_tokens": model.config.block_size
                    * (state["iter_num"] + 1)
                    * lm_config.micro_batch_size
                    * fabric.world_size,
                },
                state["step_count"],
            )
            fabric.barrier()
        if not is_accumulating and state["step_count"] % lm_config.save_step_interval == 0:
            checkpoint_path = lm_config.out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    lm_config: TinyLMConfig,
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(lm_config.eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= lm_config.eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()

    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
    split="train",
    lm_config: TinyLMConfig = None,
) -> DataLoader:
    datasets = []
    data_config = lm_config.train_data_config if split == "train" else lm_config.val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))

        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size.
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
    lm_config: TinyLMConfig = None,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
        lm_config=lm_config,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation",
            lm_config=lm_config,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, lm_config):
    # 1) linear warmup for warmup_iters steps
    if it < lm_config.warmup_iters and lm_config.need_to_warm:
        return lm_config.learning_rate * it / lm_config.warmup_iters

    # 1.5)
    if lm_config.is_constant_lr:
        # @fan if we use minicpm, WDS schedular: warmup -> stable -> Decay
        return lm_config.learning_rate

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lm_config.lr_decay_iters:
        return lm_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - lm_config.warmup_iters * lm_config.need_to_warm) / (
        lm_config.lr_decay_iters - lm_config.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

    return lm_config.min_lr + coeff * (lm_config.learning_rate - lm_config.min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
