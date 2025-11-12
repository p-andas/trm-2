#!/usr/bin/env python3
"""
Training script for TRM Attention on Sudoku-Extreme.
Auto-resumes from latest checkpoint if runpod crashes.

Checkpoints are saved to: checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/trm_att/
Checkpoint naming: step_{step_number}
"""

import os
import sys
import glob
import re
import copy
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import tqdm

import torch
import torch.distributed as dist
from torch import nn
import wandb

# Import from pretrain
from pretrain import (
    PretrainConfig, ArchConfig, LossConfig,
    load_synced_config, create_model, create_dataloader,
    init_train_state, train_batch, evaluate, save_train_state,
    save_code_and_config, load_checkpoint
)
from models.ema import EMAHelper


def find_latest_checkpoint(checkpoint_dir: str):
    """
    Find the latest checkpoint in the checkpoint directory.
    Returns: (checkpoint_path, step_number) or (None, None) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        return None, None
    
    # Find all checkpoint files matching pattern: step_XXXXX
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    
    if not checkpoint_files:
        return None, None
    
    # Extract step numbers and find the latest
    latest_step = -1
    latest_checkpoint = None
    
    for ckpt_path in checkpoint_files:
        # Extract step number from filename like "step_5000"
        match = re.search(r'step_(\d+)', os.path.basename(ckpt_path))
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = ckpt_path
    
    if latest_checkpoint:
        print(f"Found latest checkpoint: {latest_checkpoint} (step {latest_step})")
        return latest_checkpoint, latest_step
    
    return None, None


def calculate_starting_epoch(checkpoint_step: int, eval_interval: int, total_epochs: int) -> int:
    """
    Calculate which epoch to start from based on checkpoint step.
    """
    # Each eval_interval epochs = 1 iteration
    iterations_completed = checkpoint_step // eval_interval
    epochs_completed = iterations_completed * eval_interval
    
    # Add some buffer to ensure we don't skip any training
    # Round down to nearest eval_interval
    starting_epoch = (epochs_completed // eval_interval) * eval_interval
    
    return min(starting_epoch, total_epochs)


def main():
    # Configuration based on TRM paper (Section 6)
    # For Sudoku-Extreme with Attention architecture
    
    # Base checkpoint directory
    BASE_CHECKPOINT_DIR = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch"
    RUN_NAME = "trm_att"  # Base name, will append unique ID
    CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, RUN_NAME)
    
    # Check for existing checkpoints and resume
    latest_ckpt, latest_step = find_latest_checkpoint(CHECKPOINT_DIR)
    
    if latest_ckpt:
        print(f"\n{'='*80}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"Checkpoint: {latest_ckpt}")
        print(f"Step: {latest_step}")
        print(f"{'='*80}\n")
        resume_from_checkpoint = latest_ckpt
        # Calculate starting epoch (approximate)
        eval_interval = 5000
        total_epochs = 60000
        starting_epoch = calculate_starting_epoch(latest_step, eval_interval, total_epochs)
        print(f"Resuming from approximately epoch {starting_epoch} (will continue from checkpoint step {latest_step})")
    else:
        print(f"\n{'='*80}")
        print(f"STARTING NEW TRAINING")
        print(f"Checkpoint directory: {CHECKPOINT_DIR}")
        print(f"{'='*80}\n")
        resume_from_checkpoint = None
        starting_epoch = 0
    
    # Initialize distributed training if needed
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None
    
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and 
            dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )
    
    # Create config dict matching paper hyperparameters
    config_dict = {
        "arch": {
            "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
            "loss": {
                "name": "losses@ACTLossHead",
                "loss_type": "stablemax_cross_entropy"
            },
            "halt_exploration_prob": 0.1,
            "halt_max_steps": 16,
            "H_cycles": 3,
            "L_cycles": 6,
            "H_layers": 0,
            "L_layers": 2,  # 2 layers as per paper
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 4,
            "puzzle_emb_ndim": 512,
            "pos_encodings": "rope",  # Attention uses RoPE
            "forward_dtype": "bfloat16",
            "mlp_t": False,  # Use attention, not MLP
            "puzzle_emb_len": 16,
            "no_ACT_continue": True
        },
        "data_paths": ["data/sudoku-extreme-1k-aug-1000"],
        "data_paths_test": [],
        "evaluators": [],
        "global_batch_size": 768,
        "epochs": 60000,  # Paper says 60k epochs
        "lr": 1e-4,
        "lr_min_ratio": 1.0,
        "lr_warmup_steps": 2000,  # Paper says 2K iterations
        "weight_decay": 1.0,
        "beta1": 0.9,
        "beta2": 0.95,
        "puzzle_emb_lr": 1e-4,
        "puzzle_emb_weight_decay": 1.0,
        "project_name": "Sudoku-extreme-1k-aug-1000-ACT-torch",
        "run_name": RUN_NAME,
        "entity": None,
        "load_checkpoint": resume_from_checkpoint,
        "checkpoint_path": CHECKPOINT_DIR,
        "seed": 0,
        "checkpoint_every_eval": True,  # Save at every evaluation
        "checkpoint_every_n_steps": None,
        "eval_interval": 5000,  # Evaluate every 5000 epochs
        "min_eval_interval": 0,
        "eval_save_outputs": [],
        "ema": True,  # Paper uses EMA
        "ema_rate": 0.999,  # Paper says 0.999
        "freeze_weights": False
    }
    
    # Convert to OmegaConf and then to PretrainConfig
    omega_config = OmegaConf.create(config_dict)
    config = load_synced_config(omega_config, rank=RANK, world_size=WORLD_SIZE)
    
    # Override checkpoint path to ensure correct naming
    config.checkpoint_path = CHECKPOINT_DIR
    config.load_checkpoint = resume_from_checkpoint
    
    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)
    
    # Dataset setup
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."
    
    # Load metadata
    from puzzle_dataset import PuzzleDatasetMetadata
    metadata_path = os.path.join(config.data_paths[0], "metadata.json")
    train_metadata = PuzzleDatasetMetadata.model_validate_json(Path(metadata_path).read_text())
    
    # Create dataloaders
    train_loader = create_dataloader(config, "train", rank=RANK, world_size=WORLD_SIZE)
    eval_loader = create_dataloader(config, "test", rank=RANK, world_size=WORLD_SIZE)
    
    # Load eval metadata if test data exists
    eval_metadata = train_metadata
    if len(config.data_paths_test) > 0:
        eval_metadata_path = os.path.join(config.data_paths_test[0], "metadata.json")
        if os.path.exists(eval_metadata_path):
            eval_metadata = PuzzleDatasetMetadata.model_validate_json(Path(eval_metadata_path).read_text())
    
    # Create model
    model = create_model(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    
    # Initialize training state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    
    # Load checkpoint if resuming
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Loading checkpoint: {resume_from_checkpoint}")
        load_checkpoint(model, config)
        # Note: We don't load optimizer state or step count here
        # The training loop will continue from where it left off
        print("Checkpoint loaded successfully")
    
    # Initialize wandb (only on rank 0)
    if RANK == 0:
        wandb.init(
            project=config.project_name or "trm-sudoku",
            name=config.run_name,
            entity=config.entity,
            config=config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        )
    
    # Save code and config
    if RANK == 0:
        save_code_and_config(config)
    
    # EMA setup
    ema_helper = None
    if config.ema:
        ema_helper = EMAHelper(rate=config.ema_rate)
        ema_helper.register(train_state.model)
    
    # Progress bar
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=config.epochs, desc="Training")
        if latest_step is not None:
            progress_bar.update(latest_step)
    
    # Training loop
    print(f"\nStarting training loop...")
    print(f"Total iterations: {total_iters}")
    print(f"Epochs per iteration: {train_epochs_per_iter}")
    print(f"Total epochs: {config.epochs}")
    if resume_from_checkpoint:
        print(f"Resuming from step: {latest_step}")
    print()
    
    for _iter_id in range(total_iters):
        if RANK == 0:
            current_epoch = _iter_id * train_epochs_per_iter
            print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {current_epoch}/{config.epochs}")
        
        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            
            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                if progress_bar:
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            
            if config.ema:
                ema_helper.update(train_state.model)
            
            # Save checkpoint every N steps if configured
            if (
                RANK == 0
                and config.checkpoint_every_n_steps is not None
                and train_state.step % config.checkpoint_every_n_steps == 0
            ):
                save_train_state(config, train_state)
        
        # Evaluation
        if _iter_id >= config.min_eval_interval:
            if RANK == 0:
                print("EVALUATE")
            
            # Use EMA model for evaluation if enabled
            if config.ema:
                if RANK == 0:
                    print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            
            train_state_eval.model.eval()
            metrics = evaluate(
                config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                config.evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP
            )
            
            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
            
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                # Save with EMA model if using EMA
                save_train_state(config, train_state_eval)
                print(f"Checkpoint saved: step_{train_state.step}")
            
            if config.ema:
                del train_state_eval
    
    # Finalize
    if RANK == 0:
        print(f"\nTraining completed!")
        print(f"Final checkpoint saved at: {CHECKPOINT_DIR}/step_{train_state.step}")
        if progress_bar:
            progress_bar.close()
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if RANK == 0:
        wandb.finish()


if __name__ == "__main__":
    main()

