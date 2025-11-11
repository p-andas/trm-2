import sys
sys.path.insert(0, "/workspace/TinyRecursiveModels")

import torch
import numpy as np
from tqdm import tqdm
import yaml

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def main():
    print("=" * 80)
    print("SUDOKU-EXTREME TEST SET EVALUATION")
    print("=" * 80)

    # Paths
    CHECKPOINT_PATH = "/workspace/TinyRecursiveModels/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/sudoku_mlp_h100_test/step_78120"
    CONFIG_PATH = "/workspace/TinyRecursiveModels/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/sudoku_mlp_h100_test/all_config.yaml"
    TEST_DATA_DIR = "/workspace/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000/test"

    print("\n1. Loading configuration...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    print("   ✓ Config loaded")

    print("\n2. Loading test data...")
    test_inputs = np.load(f"{TEST_DATA_DIR}/all__inputs.npy")
    test_labels = np.load(f"{TEST_DATA_DIR}/all__labels.npy")
    print(f"   ✓ Test examples: {len(test_inputs):,}")

    print("\n3. Loading model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    # Detect vocab_size and num_puzzle_identifiers from checkpoint
    embed_key = [k for k in checkpoint.keys() if 'embed_tokens.embedding_weight' in k][0]
    puzzle_emb_key = [k for k in checkpoint.keys() if 'puzzle_emb.weights' in k][0]
    vocab_size = checkpoint[embed_key].shape[0]
    num_puzzle_identifiers = checkpoint[puzzle_emb_key].shape[0]

    # Model config
    model_config = {
        'batch_size': config['global_batch_size'],
        'seq_len': 81,
        'num_puzzle_identifiers': num_puzzle_identifiers,
        'vocab_size': vocab_size,
        'hidden_size': config['arch']['hidden_size'],
        'H_cycles': config['arch']['H_cycles'],
        'L_cycles': config['arch']['L_cycles'],
        'H_layers': config['arch']['H_layers'],
        'L_layers': config['arch']['L_layers'],
        'mlp_t': config['arch']['mlp_t'],
        'pos_encodings': config['arch']['pos_encodings'],
        'halt_max_steps': config['arch']['halt_max_steps'],
        'expansion': config['arch']['expansion'],
        'num_heads': config['arch']['num_heads'],
        'forward_dtype': config['arch']['forward_dtype'],
        'halt_exploration_prob': config['arch']['halt_exploration_prob'],
        'puzzle_emb_len': config['arch']['puzzle_emb_len'],
        'puzzle_emb_ndim': config['arch']['puzzle_emb_ndim'],
        'no_ACT_continue': config['arch'].get('no_ACT_continue', True),
    }

    model = TinyRecursiveReasoningModel_ACTV1(model_config)

    # Load checkpoint weights
    clean_state_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('_orig_mod.model.', '').replace('module.', '')
        clean_state_dict[new_k] = v

    model.load_state_dict(clean_state_dict, strict=False)

    # Move to GPU
    device = torch.device('cuda')
    model = model.to(device)
    if hasattr(model.inner, 'H_init'):
        model.inner.H_init = model.inner.H_init.to(device)
    if hasattr(model.inner, 'L_init'):
        model.inner.L_init = model.inner.L_init.to(device)

    model.eval()
    print(f"   ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters)")

    print(f"\n4. Running inference on {len(test_inputs):,} test puzzles...")

    # Evaluation loop
    batch_size = 256
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_inputs), batch_size), desc="Evaluating"):
            batch_inputs = torch.from_numpy(test_inputs[i:i+batch_size]).long().cuda()
            batch_labels = torch.from_numpy(test_labels[i:i+batch_size]).long().cuda()
            
            batch = {
                'inputs': batch_inputs,
                'labels': batch_labels,
                'puzzle_identifiers': torch.zeros(len(batch_inputs), dtype=torch.long, device='cuda'),
            }
            
            with torch.cuda.device('cuda:0'):
                carry = model.initial_carry(batch)
                if hasattr(carry, 'inner_carry'):
                    if hasattr(carry.inner_carry, 'z_H') and carry.inner_carry.z_H.device.type != 'cuda':
                        carry.inner_carry.z_H = carry.inner_carry.z_H.cuda()
                    if hasattr(carry.inner_carry, 'z_L') and carry.inner_carry.z_L.device.type != 'cuda':
                        carry.inner_carry.z_L = carry.inner_carry.z_L.cuda()
                if hasattr(carry, 'steps') and carry.steps.device.type != 'cuda':
                    carry.steps = carry.steps.cuda()
                if hasattr(carry, 'halted') and carry.halted.device.type != 'cuda':
                    carry.halted = carry.halted.cuda()
            
            max_steps = 16 
        
            for step in range(max_steps):
                result = model(carry=carry, batch=batch)
                
                if len(result) == 2:
                    carry, outputs = result
                elif len(result) == 5:
                    carry, loss, metrics, outputs, all_finish = result
                else:
                    raise ValueError(f"Unexpected return format: {len(result)} values")
            
            if 'logits' in outputs:
                preds = torch.argmax(outputs['logits'], dim=-1)
            elif 'preds' in outputs:
                preds = outputs['preds']
            else:
                raise ValueError(f"Cannot find predictions. Available: {outputs.keys()}")
            
            exact_matches = (preds == batch_labels).all(dim=1)
            num_correct += exact_matches.sum().item()
            num_total += len(batch_inputs)

    test_accuracy = 100.0 * num_correct / num_total

    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Test examples: {num_total:,}")
    print(f"Correctly solved: {num_correct:,}")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"\nPaper baseline (TRM-MLP): 87.4%")
    print(f"Our model: {test_accuracy:.2f}%")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
