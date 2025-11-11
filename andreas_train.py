import os
import glob

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    #find most recent checkpoint
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "**", "*.pt"), recursive=True), key=os.path.getmtime)
    if checkpoints:
        return checkpoints[-1]
    return None

def main():
    # try resuming from latest checkpoint if exists
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        print(f"\nResuming from latest checkpoint: {latest_ckpt}\n")
        resume_flag = f"+load_checkpoint={latest_ckpt}"
    else:
        print("\nNo checkpoint found, starting new training run.\n")
        resume_flag = ""

    # Training command
    command = (
        "python pretrain.py "
        "arch=trm_singlez "  # single-z model used in TRM paper
        "data_paths='[data/sudoku-extreme-1k-aug-1000]' "
        "evaluators='[]' "
        "epochs=60000 "
        "eval_interval=5000 "
        "lr=1e-4 "
        "puzzle_emb_lr=1e-4 "
        "weight_decay=1.0 "
        "puzzle_emb_weight_decay=1.0 "
        "beta1=0.9 "
        "beta2=0.95 "
        "lr_warmup_steps=2000 "
        "global_batch_size=768 "
        "arch.mlp_t=True "
        "arch.pos_encodings=none "
        "arch.L_layers=2 "
        "arch.H_cycles=3 "
        "arch.L_cycles=6 "
        "arch.hidden_size=512 "
        "arch.expansion=4 "
        "arch.num_heads=8 "
        "ema=True "
        "ema_rate=0.999 "
        f"{resume_flag} "
        "+run_name=sudoku_trm_singlez"
    )

    # Run training
    os.system(command)

    print("\nTRM-SingleZ training started (or resumed) on Sudoku dataset.")
    print("Checkpoints are saved under ./checkpoints/")
    print("If training crashes, just rerun this script — it will auto-resume from the last checkpoint.")

if __name__ == "__main__":
    main()
