import os
 
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
 
import json
import torch
from tqdm.auto import tqdm
 
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    TopKTrainingSAEConfig,
    LoggingConfig,
)
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
LAYER = 5
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"
 
D_MODEL = 768       
D_SAE = D_MODEL * 8   
K = 50                 
 
TRAINING_TOKENS = int(10e6)   
TRAIN_BATCH_SIZE_TOKENS = 4096
CONTEXT_SIZE = 128
 
SEEDS = [0, 1, 2, 3, 4]   

MODEL_CONDITIONS = {
    "baseline_gpt2":        "gpt2",                          # pretrained base, no fine-tuning
    "full_ft":              "./gpt2-cola-full",
    "lora_r4":               "./gpt2-cola-lora-r4-merged",
    "lora_r8":               "./gpt2-cola-lora-r8-merged",
    "lora_r16":              "./gpt2-cola-lora-r16-merged",
    "lora_r32":              "./gpt2-cola-lora-r32-merged",
}

SAE_TRAINING_DATASET = "Skylion007/openwebtext"
 
CHECKPOINT_ROOT = "./sae_checkpoints"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
 

def train_one_sae(model_name_or_path: str, seed: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
 
    cfg = LanguageModelSAERunnerConfig(
        model_name=model_name_or_path,
        hook_name=HOOK_NAME,
        dataset_path=SAE_TRAINING_DATASET,
        is_dataset_tokenized=False,
        streaming=True,
        context_size=CONTEXT_SIZE,
        training_tokens=TRAINING_TOKENS,
        train_batch_size_tokens=TRAIN_BATCH_SIZE_TOKENS,
        lr=1e-4,
        lr_scheduler_name="constant",
        adam_beta1=0.9,
        adam_beta2=0.999,
        n_batches_in_buffer=32,
        store_batch_size_prompts=16,
        logger=LoggingConfig(log_to_wandb=False),
        n_checkpoints=0,             # only save final SAE, no intermediate checkpoints
        checkpoint_path=save_dir,
        device=device,
        seed=seed,
        sae=TopKTrainingSAEConfig(
            d_in=D_MODEL,
            d_sae=D_SAE,
            k=K,
            normalize_activations="expected_average_only_in",
        ),
    )
 
    runner = LanguageModelSAETrainingRunner(cfg)
    sae = runner.run()
    return sae
 
def already_trained(save_dir: str) -> bool:
    cfg_file = os.path.join(save_dir, "cfg.json")
    return os.path.exists(cfg_file)
 
 
manifest = {}
 
total_runs = len(MODEL_CONDITIONS) * len(SEEDS)
pbar = tqdm(total=total_runs, desc="Training SAEs (layer 3, all conditions/seeds)")
 
for condition, model_path in MODEL_CONDITIONS.items():
    for seed in SEEDS:
        save_dir = os.path.join(CHECKPOINT_ROOT, condition, f"layer{LAYER}_seed{seed}")
 
        if already_trained(save_dir):
            pbar.write(f"[skip] {condition} | seed {seed} | already trained -> {save_dir}")
            pbar.update(1)
            manifest.setdefault(condition, []).append(save_dir)
            continue
 
        pbar.write(f"[train] {condition} | seed {seed} | model={model_path}")
        try:
            train_one_sae(model_path, seed, save_dir)
            manifest.setdefault(condition, []).append(save_dir)
        except Exception as e:
            pbar.write(f"[ERROR] {condition} | seed {seed} failed: {e}")
        pbar.update(1)
 
pbar.close()
 

manifest_path = os.path.join(CHECKPOINT_ROOT, "manifest_layer3.json")
with open(manifest_path, "w") as f:
    json.dump(
        {
            "layer": LAYER,
            "hook_name": HOOK_NAME,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "k": K,
            "training_tokens": TRAINING_TOKENS,
            "seeds": SEEDS,
            "conditions": manifest,
        },
        f,
        indent=2,
    )
 
print(f"\nDone. Manifest saved to {manifest_path}")
print("Conditions trained:")
for cond, paths in manifest.items():
    print(f"  {cond}: {len(paths)} SAE(s)")
 
