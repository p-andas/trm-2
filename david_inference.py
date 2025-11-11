import torch
import torch.nn as nn
from pathlib import Path
import sys
from tqdm import tqdm
import json
from omegaconf import OmegaConf
from typing import List, Dict, Any
import os

# -----------------------------------------------------------------------------
# Set working directory and imports
# -----------------------------------------------------------------------------
os.chdir('/workspace/TinyRecursiveModels')
sys.path.insert(0, '/workspace/TinyRecursiveModels')

from pretrain import create_model, create_dataloader
from config import PretrainConfig
from dataset.sudoku_dataset import PuzzleDatasetMetadata
from models.ema import EMAHelper


def load_model_from_checkpoint(checkpoint_path: str, use_ema: bool = True):
    """
    Load model exactly as they do, including EMA handling
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Get config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        if hasattr(config_dict, '__dict__'):
            config_dict = config_dict.__dict__
        config = PretrainConfig(**config_dict)
    else:
        raise ValueError("No config in checkpoint")
    
    # -----------------------------------------------------------------------------
    # ✅ MODIFICATION 2: Explicit dataset path
    # -----------------------------------------------------------------------------
    data_path = "/workspace/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000"
    
    metadata_path = Path(data_path) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata at: {metadata_path}")
    
    metadata = PuzzleDatasetMetadata.model_validate_json(metadata_path.read_text())
    
    print(f"\nDataset metadata:")
    print(f"  Vocab size: {metadata.vocab_size}")
    print(f"  Max seq len: {metadata.max_seq_len}")
    print(f"  Puzzle type: {metadata.puzzle_type}")
    
    # Create model using their exact function
    model, optimizers, optimizer_lrs = create_model(
        config,
        metadata,
        rank=0,
        world_size=1
    )
    
    model = model.cuda()
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Handle EMA if requested
    if use_ema and config.ema:
        print("\n✓ Using EMA weights")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(model)
        
        # Load EMA state if available
        if 'ema_state_dict' in checkpoint:
            ema_helper.load_state_dict(checkpoint['ema_state_dict'])
        
        # Apply EMA weights to model
        ema_helper.ema(model)
    else:
        print("\n✓ Using regular model weights")
    
    model.eval()
    
    return model, config, metadata


def evaluate_model(
    model: nn.Module,
    data_path: str,
    eval_metadata: PuzzleDatasetMetadata,
    config: PretrainConfig,
    split: str = 'test',
    max_inference_steps: int = 100
):
    """
    Evaluate using their exact procedure from evaluate() function
    """
    print(f"\n{'='*70}")
    print(f"Evaluating on {split} split")
    print(f"{'='*70}\n")
    
    # Create dataloader using their function
    eval_loader = create_dataloader(
        config,
        split=split,
        rank=0,
        world_size=1,
        shuffle=False
    )
    
    all_metrics = {}
    processed_batches = 0
    
    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            print(f"Processing batch {processed_batches}: {set_name}")
            
            batch = {k: v.cuda() for k, v in batch.items()}
            
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            inference_steps = 0
            while inference_steps < max_inference_steps:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_keys=set()
                )
                inference_steps += 1
                
                if all_finish:
                    break
            
            print(f"  Completed inference in {inference_steps} steps")
            
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v.item())
            
            del carry, loss, preds, batch, all_finish
    
    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    
    print(f"\n{'='*70}")
    print(f"RESULTS ({split}):")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Total batches: {processed_batches}")
    print(f"{'='*70}\n")
    
    return avg_metrics


def main():
    """
    Main evaluation function
    """
    import argparse
    parser = argparse.ArgumentParser()
    
    # -----------------------------------------------------------------------------
    # MODIFICATION 1: Default checkpoint path to your final trained model
    # -----------------------------------------------------------------------------
    parser.add_argument(
        '--checkpoint',
        default='/workspace/TinyRecursiveModels/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/sudoku_mlp_h100_test/step_78120',
        help='Path to checkpoint'
    )
    
    parser.add_argument('--split', default='test', help='Dataset split to evaluate')
    parser.add_argument('--use_ema', action='store_true', default=True, help='Use EMA weights')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false', help='Do not use EMA')
    parser.add_argument('--max_steps', type=int, default=100, help='Max inference steps')
    parser.add_argument('--output', default='results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    model, config, metadata = load_model_from_checkpoint(
        args.checkpoint,
        use_ema=args.use_ema
    )
    
    print(f"\nModel: {config.arch.name}")
    print(f"d_model: {config.arch.d_model}")
    
    # -----------------------------------------------------------------------------
    # Use the correct dataset path again here
    # -----------------------------------------------------------------------------
    data_path = "/workspace/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000"
    
    results = evaluate_model(
        model,
        data_path,
        metadata,
        config,
        split=args.split,
        max_inference_steps=args.max_steps
    )
    
    output_data = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'use_ema': args.use_ema,
        'max_inference_steps': args.max_steps,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    return results


if __name__ == '__main__':
    main()
