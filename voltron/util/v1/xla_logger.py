"""
xla_logger.py

Utility class defining various XLA logging methods (called within marked closures), for logging metrics periodically
through training & validation.
"""
from typing import List

import jsonlines
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import wandb


# === Generic (Cross-Model) Epoch End Update ===
def log_epoch_end_update(
    arch: str,
    epoch: int,
    global_step: int,
    run_id: str,
    duration: int,
    train_losses: List[torch.Tensor],
    val_loss: float,
    lr: float,
    step_times: List[float],
) -> None:
    train_loss = torch.stack(list(train_losses)).mean()
    average_step_time = np.mean(list(step_times))

    # Console Logging --> Unclear if it'll work?
    xm.master_print(
        f"Epoch {epoch:03d}, Global Step {global_step:06d} || LR :: {lr:.6f} -- Train Loss :: {train_loss:.4f} "
        f"-- Val Loss :: {val_loss:.4f} -- Total Time (sec) :: {duration}"
    )

    # Get Log-Friendly Arch
    p_arch = {
        "v-mvp": "MVP",
        "v-r3m": "R3M (ViT)",
        "v-rn3m": "R3M (RN)",
        "v-cond": "V-Cond",
        "v-dual": "V-Dual",
        "v-gen": "V-Gen",
    }[arch]

    # Log to Weights & Biases & JSONL
    blob = {
        "Pretrain/Step": global_step,
        "Pretrain/Epoch": epoch,
        "Pretrain/Training Duration": duration,
        "Pretrain/Step Time": average_step_time,
        f"Pretrain/{p_arch} Train Epoch Loss": train_loss.item(),
        f"Pretrain/{p_arch} Train Loss": train_loss.item(),
        f"Pretrain/{p_arch} Validation Loss": val_loss,
        "Pretrain/Learning Rate": lr,
    }

    wandb.log(blob, step=global_step)
    with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
        js_logger.write(blob)


# === Data-Locked Reproductions ===


def log_vmvp_train_update(
    epoch: int,
    global_step: int,
    run_id: str,
    train_losses: List[torch.Tensor],
    lr: float,
    reconstruction_losses: List[torch.Tensor],
    step_times: List[float],
) -> None:
    train_loss = torch.stack(list(train_losses)).mean()
    reconstruction_loss = torch.stack(list(reconstruction_losses)).mean()
    average_step_time = np.mean(list(step_times))

    # Console Logging --> Just log the aggregated train loss...
    xm.master_print(
        f"Epoch {epoch:03d}, Global Step {global_step:06d} || LR :: {lr:.6f} -- Train Loss :: {train_loss:.4f}"
    )

    # Log to Weights & Biases + JSONL
    blob = {
        "Pretrain/Step": global_step,
        "Pretrain/Epoch": epoch,
        "Pretrain/V-MVP Train Loss": train_loss.item(),
        "Pretrain/Reconstruction Loss": reconstruction_loss.item(),
        "Pretrain/Learning Rate": lr,
        "Pretrain/Step Time": average_step_time,
    }
    wandb.log(blob, step=global_step)
    with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
        js_logger.write(blob)


def log_vr3m_train_update(
    epoch: int,
    global_step: int,
    run_id: str,
    train_losses: List[torch.Tensor],
    lr: float,
    tcn_losses: List[torch.Tensor],
    reward_losses: List[torch.Tensor],
    l1_losses: List[torch.Tensor],
    l2_losses: List[torch.Tensor],
    tcn_accuracies: List[torch.Tensor],
    reward_accuracies: List[torch.Tensor],
    step_times: List[float],
) -> None:
    train_loss = torch.stack(list(train_losses)).mean()
    tcn_loss = torch.stack(list(tcn_losses)).mean()
    reward_loss = torch.stack(list(reward_losses)).mean()
    l1_loss, l2_loss = torch.stack(list(l1_losses)).mean(), torch.stack(list(l2_losses)).mean()
    tcn_accuracy = torch.stack(list(tcn_accuracies)).mean()
    reward_accuracy = torch.stack(list(reward_accuracies)).mean()
    average_step_time = np.mean(list(step_times))

    # Console Logging --> Just log the aggregated train loss...
    xm.master_print(
        f"Epoch {epoch:03d}, Global Step {global_step:06d} || LR :: {lr:.6f} -- Train Loss :: {train_loss:.4f}"
    )

    # Log to Weights & Biases + JSONL
    blob = {
        "Pretrain/Step": global_step,
        "Pretrain/Epoch": epoch,
        "Pretrain/V-R3M Train Loss": train_loss.item(),
        "Pretrain/TCN Loss": tcn_loss.item(),
        "Pretrain/Reward Loss": reward_loss.item(),
        "Pretrain/L1 Loss": l1_loss.item(),
        "Pretrain/L2 Loss": l2_loss.item(),
        "Pretrain/TCN Accuracy": tcn_accuracy.item(),
        "Pretrain/Reward Accuracy": reward_accuracy.item(),
        "Pretrain/Learning Rate": lr,
        "Pretrain/Step Time": average_step_time,
    }
    wandb.log(blob, step=global_step)
    with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
        js_logger.write(blob)


def log_vrn3m_train_update(
    epoch: int,
    global_step: int,
    run_id: str,
    train_losses: List[torch.Tensor],
    lr: float,
    tcn_losses: List[torch.Tensor],
    reward_losses: List[torch.Tensor],
    l1_losses: List[torch.Tensor],
    l2_losses: List[torch.Tensor],
    tcn_accuracies: List[torch.Tensor],
    reward_accuracies: List[torch.Tensor],
    step_times: List[float],
) -> None:
    train_loss = torch.stack(list(train_losses)).mean()
    tcn_loss = torch.stack(list(tcn_losses)).mean()
    reward_loss = torch.stack(list(reward_losses)).mean()
    l1_loss, l2_loss = torch.stack(list(l1_losses)).mean(), torch.stack(list(l2_losses)).mean()
    tcn_accuracy = torch.stack(list(tcn_accuracies)).mean()
    reward_accuracy = torch.stack(list(reward_accuracies)).mean()
    average_step_time = np.mean(list(step_times))

    # Console Logging --> Just log the aggregated train loss...
    xm.master_print(
        f"Epoch {epoch:03d}, Global Step {global_step:06d} || LR :: {lr:.6f} -- Train Loss :: {train_loss:.4f}"
    )

    # Log to Weights & Biases + JSONL
    blob = {
        "Pretrain/Step": global_step,
        "Pretrain/Epoch": epoch,
        "Pretrain/V-RN3M Train Loss": train_loss.item(),
        "Pretrain/TCN Loss": tcn_loss.item(),
        "Pretrain/Reward Loss": reward_loss.item(),
        "Pretrain/L1 Loss": l1_loss.item(),
        "Pretrain/L2 Loss": l2_loss.item(),
        "Pretrain/TCN Accuracy": tcn_accuracy.item(),
        "Pretrain/Reward Accuracy": reward_accuracy.item(),
        "Pretrain/Learning Rate": lr,
        "Pretrain/Step Time": average_step_time,
    }
    wandb.log(blob, step=global_step)
    with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
        js_logger.write(blob)


# === Voltron Models ===
def log_vcond_train_update(
    epoch: int,
    global_step: int,
    run_id: str,
    train_losses: List[torch.Tensor],
    lr: float,
    reconstruction_losses: List[torch.Tensor],
    step_times: List[float],
) -> None:
    train_loss = torch.stack(list(train_losses)).mean()
    reconstruction_loss = torch.stack(list(reconstruction_losses)).mean()
    average_step_time = np.mean(list(step_times))

    # Console Logging --> Just log the aggregated train loss...
    xm.master_print(
        f"Epoch {epoch:03d}, Global Step {global_step:06d} || LR :: {lr:.6f} -- Train Loss :: {train_loss:.4f}"
    )

    # Log to Weights & Biases + JSONL
    blob = {
        "Pretrain/Step": global_step,
        "Pretrain/Epoch": epoch,
        "Pretrain/V-Cond Train Loss": train_loss.item(),
        "Pretrain/Reconstruction Loss": reconstruction_loss.item(),
        "Pretrain/Learning Rate": lr,
        "Pretrain/Step Time": average_step_time,
    }
    wandb.log(blob, step=global_step)
    with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
        js_logger.write(blob)


def log_vdual_train_update(
    epoch: int,
    global_step: int,
    run_id: str,
    train_losses: List[torch.Tensor],
    lr: float,
    reconstruction_losses: List[torch.Tensor],
    zero_reconstruction_losses: List[torch.Tensor],
    k_reconstruction_losses: List[torch.Tensor],
    step_times: List[float],
) -> None:
    train_loss = torch.stack(list(train_losses)).mean()
    reconstruction_loss = torch.stack(list(reconstruction_losses)).mean()
    zero_reconstruction_loss = torch.stack(list(zero_reconstruction_losses)).mean()
    k_reconstruction_loss = torch.stack(list(k_reconstruction_losses)).mean()
    average_step_time = np.mean(list(step_times))

    # Console Logging --> Just log the aggregated train loss...
    xm.master_print(
        f"Epoch {epoch:03d}, Global Step {global_step:06d} || LR :: {lr:.6f} -- Train Loss :: {train_loss:.4f}"
    )

    # Log to Weights & Biases + JSONL
    blob = {
        "Pretrain/Step": global_step,
        "Pretrain/Epoch": epoch,
        "Pretrain/V-Dual Train Loss": train_loss.item(),
        "Pretrain/Reconstruction Loss": reconstruction_loss.item(),
        "Pretrain/Zero Reconstruction Loss": zero_reconstruction_loss.item(),
        "Pretrain/K Reconstruction Loss": k_reconstruction_loss.item(),
        "Pretrain/Learning Rate": lr,
        "Pretrain/Step Time": average_step_time,
    }
    wandb.log(blob, step=global_step)
    with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
        js_logger.write(blob)


def log_vgen_train_update(
    epoch: int,
    global_step: int,
    run_id: str,
    train_losses: List[torch.Tensor],
    lr: float,
    reconstruction_losses: List[torch.Tensor],
    lm_losses: List[torch.Tensor],
    lm_ppl: List[torch.Tensor],
    zero_reconstruction_losses: List[torch.Tensor],
    k_reconstruction_losses: List[torch.Tensor],
    step_times: List[float],
) -> None:
    train_loss = torch.stack(list(train_losses)).mean()
    reconstruction_loss = torch.stack(list(reconstruction_losses)).mean()
    lm_loss = torch.stack(list(lm_losses)).mean()
    lm_perplexity = torch.stack(list(lm_ppl)).mean()
    zero_reconstruction_loss = torch.stack(list(zero_reconstruction_losses)).mean()
    k_reconstruction_loss = torch.stack(list(k_reconstruction_losses)).mean()
    average_step_time = np.mean(list(step_times))

    # Console Logging --> Just log the aggregated train loss...
    xm.master_print(
        f"Epoch {epoch:03d}, Global Step {global_step:06d} || LR :: {lr:.6f} -- Train Loss :: {train_loss:.4f} --"
        f" Reconstruction Loss {reconstruction_loss:.4f} -- LM Loss {lm_loss:.4f}"
    )

    # Log to Weights & Biases + JSONL
    blob = {
        "Pretrain/Step": global_step,
        "Pretrain/Epoch": epoch,
        "Pretrain/V-Gen Train Loss": train_loss.item(),
        "Pretrain/Reconstruction Loss": reconstruction_loss.item(),
        "Pretrain/CLM Loss": lm_loss.item(),
        "Pretrain/CLM Perplexity": lm_perplexity.item(),
        "Pretrain/LM Loss": lm_loss.item(),
        "Pretrain/LM Perplexity": lm_perplexity.item(),
        "Pretrain/Zero Reconstruction Loss": zero_reconstruction_loss.item(),
        "Pretrain/K Reconstruction Loss": k_reconstruction_loss.item(),
        "Pretrain/Learning Rate": lr,
        "Pretrain/Step Time": average_step_time,
    }
    wandb.log(blob, step=global_step)
    with jsonlines.open(f"{run_id}.jsonl", mode="a", sort_keys=True) as js_logger:
        js_logger.write(blob)
