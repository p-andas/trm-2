# andreas_inference.py
import sys
sys.path.insert(0, "/workspace/TinyRecursiveModels")

import torch
import numpy as np
import yaml
from tqdm import tqdm
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


# ------------------------------- PATHS ---------------------------------------
CHECKPOINT_PATH = (
    "/workspace/TinyRecursiveModels/checkpoints/"
    "Sudoku-extreme-1k-aug-1000-ACT-torch/sudoku_mlp_h100_test/step_78120"
)
CONFIG_PATH = (
    "/workspace/TinyRecursiveModels/checkpoints/"
    "Sudoku-extreme-1k-aug-1000-ACT-torch/sudoku_mlp_h100_test/all_config.yaml"
)
TEST_DATA_DIR = "/workspace/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000/test"


print("=" * 80)
print("SUDOKU-EXTREME TEST SET EVALUATION (TRM-INFER)")
print("=" * 80)

# ------------------------------- CONFIG --------------------------------------
print("\n1. Loading configuration...")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
print("   ✓ Config loaded")

# ------------------------------- DATA ----------------------------------------
print("\n2. Loading test data...")
test_inputs = np.load(f"{TEST_DATA_DIR}/all__inputs.npy")
test_labels = np.load(f"{TEST_DATA_DIR}/all__labels.npy")
puzzle_identifiers = np.load(f"{TEST_DATA_DIR}/all__puzzle_identifiers.npy")
print(f"   ✓ Loaded {len(test_inputs):,} test puzzles")

# ------------------------------- CHECKPOINT ----------------------------------
print("\n3. Loading model checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
print(f"   ✓ Model loaded ({sum(v.numel() for v in checkpoint.values())/1e6:.2f}M params)")

# infer sizes from checkpoint
embed_key = [k for k in checkpoint.keys() if "embed_tokens.embedding_weight" in k][0]
puzzle_emb_key = [k for k in checkpoint.keys() if "puzzle_emb.weights" in k][0]
vocab_size = checkpoint[embed_key].shape[0]
num_puzzle_identifiers = checkpoint[puzzle_emb_key].shape[0]

# ------------------------------- MODEL CONFIG --------------------------------
model_config = {
    "batch_size": config["global_batch_size"],
    "seq_len": 81,
    "num_puzzle_identifiers": num_puzzle_identifiers,
    "vocab_size": vocab_size,
    "hidden_size": config["arch"]["hidden_size"],
    "H_cycles": config["arch"]["H_cycles"],
    "L_cycles": config["arch"]["L_cycles"],
    "H_layers": config["arch"]["H_layers"],
    "L_layers": config["arch"]["L_layers"],
    "mlp_t": config["arch"]["mlp_t"],
    "pos_encodings": config["arch"]["pos_encodings"],
    "halt_max_steps": 16,
    "expansion": config["arch"]["expansion"],
    "num_heads": config["arch"]["num_heads"],
    "forward_dtype": config["arch"]["forward_dtype"],
    "halt_exploration_prob": config["arch"]["halt_exploration_prob"],
    "puzzle_emb_len": config["arch"]["puzzle_emb_len"],
    "puzzle_emb_ndim": config["arch"]["puzzle_emb_ndim"],
    "no_ACT_continue": config["arch"].get("no_ACT_continue", True),
}

# ------------------------------- INIT MODEL ----------------------------------
model = TinyRecursiveReasoningModel_ACTV1(model_config)

# clean checkpoint keys (remove wrappers like _orig_mod.model. / module.)
clean_state_dict = {}
for k, v in checkpoint.items():
    new_k = k.replace("_orig_mod.model.", "").replace("module.", "")
    clean_state_dict[new_k] = v
model.load_state_dict(clean_state_dict, strict=False)

# ------------------------------- DEVICE SETUP --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n> Using device: {device.type}")
model.to(device)
model.eval()

# IMPORTANT: some custom buffers don't always move with .to(device)
if hasattr(model, "inner"):
    if hasattr(model.inner, "H_init"):
        model.inner.H_init = model.inner.H_init.to(device)
    if hasattr(model.inner, "L_init"):
        model.inner.L_init = model.inner.L_init.to(device)

# helper to move the nested carry dataclasses to device
def carry_to_device(c):
    # TinyRecursiveReasoningModel_ACTV1Carry holds: inner_carry, steps, halted, current_data
    if hasattr(c, "inner_carry"):
        ic = c.inner_carry
        if hasattr(ic, "z_H"):
            ic.z_H = ic.z_H.to(device)
        if hasattr(ic, "z_L"):
            ic.z_L = ic.z_L.to(device)
        c.inner_carry = ic
    if hasattr(c, "steps"):
        c.steps = c.steps.to(device)
    if hasattr(c, "halted"):
        c.halted = c.halted.to(device)
    if hasattr(c, "current_data"):
        c.current_data = {k: v.to(device) for k, v in c.current_data.items()}
    return c

# ------------------------------- INFERENCE -----------------------------------
print(f"\n4. Running inference (dtype={config['arch']['forward_dtype']})...")
batch_size = 256
num_correct = 0
num_total = 0

with torch.no_grad():
    for i in tqdm(range(0, len(test_inputs), batch_size), desc="Evaluating", ncols=100):
        batch_inputs = torch.from_numpy(test_inputs[i:i + batch_size]).long().to(device)
        batch_labels = torch.from_numpy(test_labels[i:i + batch_size]).long().to(device)
        batch_identifiers = torch.from_numpy(puzzle_identifiers[i:i + batch_size]).long().to(device)

        batch = {
            "inputs": batch_inputs,
            "labels": batch_labels,
            "puzzle_identifiers": batch_identifiers,
        }

        # initial_carry creates CPU tensors; move the whole structure to GPU
        carry = model.initial_carry(batch)
        carry = carry_to_device(carry)

        # do max 16 steps (evaluation uses fixed steps for batching)
        for _ in range(16):
            result = model(carry=carry, batch=batch)
            carry, outputs = result[:2]  # (carry, outputs, ...) for training; we only need first two
            # the returned carry inherits device of inputs; no extra move needed

        logits = outputs.get("logits", None)
        preds = torch.argmax(logits, dim=-1) if logits is not None else outputs["preds"]

        # exact match (all 81 digits correct)
        exact_matches = (preds == batch_labels).all(dim=1)
        num_correct += exact_matches.sum().item()
        num_total += batch_inputs.size(0)

# ------------------------------- RESULTS -------------------------------------
test_accuracy = 100.0 * num_correct / num_total
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Total test puzzles: {num_total:,}")
print(f"Correctly solved:  {num_correct:,}")
print(f"Test accuracy:     {test_accuracy:.2f}%")
print("=" * 80)
print("Paper baseline (TRM-MLP): 87.4%")
print(f"Our model: {test_accuracy:.2f}%")
print("=" * 80)
