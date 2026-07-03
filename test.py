import os
 
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
 
import json
import torch
from transformers import GPT2Model, AutoModelForCausalLM, AutoModelForSequenceClassification
 
device = "cpu"  # weight comparison is cheap, CPU is fine
 
# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BASE_MODEL = "gpt2"
 
FT_MODELS = {
    "full_ft":  "./gpt2-cola-full",
    "lora_r4":  "./gpt2-cola-lora-r4-merged",
    "lora_r8":  "./gpt2-cola-lora-r8-merged",
    "lora_r16": "./gpt2-cola-lora-r16-merged",
    "lora_r32": "./gpt2-cola-lora-r32-merged",
}
 
# Layers we care about for the SAE experiment
TARGET_LAYERS = [2, 6, 10]
 
# Also report the specific layer used first in the SAE runs
PRIMARY_LAYER = 3
 
ALL_LAYERS = sorted(set(TARGET_LAYERS + [PRIMARY_LAYER]))
 
# Which weight matrices within a block to inspect
# (GPT-2 block: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj)
SUBMODULES = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
 
SAVE_DIR = "./diagnostics"
os.makedirs(SAVE_DIR, exist_ok=True)
 
 
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_transformer_body(path_or_name: str) -> GPT2Model:
    """
    Load just the GPT-2 transformer body (GPT2Model) regardless of whether
    the checkpoint is a base LM, a CausalLM, or a SequenceClassification model.
    We read the underlying transformer.* weights so all conditions are
    directly comparable.
    """
    # Try plain GPT2Model first (works for "gpt2" and any HF dir containing
    # a compatible transformer body).
    try:
        return GPT2Model.from_pretrained(path_or_name).to(device).eval()
    except Exception:
        pass
 
    # Fall back: load a wrapper model and grab .transformer
    for loader in (AutoModelForSequenceClassification, AutoModelForCausalLM):
        try:
            m = loader.from_pretrained(path_or_name).to(device).eval()
            if hasattr(m, "transformer"):
                return m.transformer
        except Exception:
            continue
 
    raise RuntimeError(f"Could not load a GPT-2 transformer body from {path_or_name}")
 
 
def get_block_weights(body: GPT2Model, layer: int):
    """Return dict of {submodule_name: weight_tensor} for a given block."""
    block = body.h[layer]
    weights = {}
    for name in SUBMODULES:
        mod = block
        for part in name.split("."):
            mod = getattr(mod, part)
        weights[name] = mod.weight.detach().float()
    return weights
 
 
def rel_frobenius_change(w_base: torch.Tensor, w_ft: torch.Tensor) -> float:
    num = torch.linalg.norm(w_ft - w_base).item()
    den = torch.linalg.norm(w_base).item()
    return num / max(den, 1e-12)
 
 
# ----------------------------------------------------------------------
# Load base once
# ----------------------------------------------------------------------
print(f"Loading base model: {BASE_MODEL}")
base_body = load_transformer_body(BASE_MODEL)
base_weights = {layer: get_block_weights(base_body, layer) for layer in ALL_LAYERS}
 
 
# ----------------------------------------------------------------------
# Compare each fine-tuned model
# ----------------------------------------------------------------------
results = {}  # condition -> layer -> submodule -> rel_change
 
for cond, path in FT_MODELS.items():
    if not os.path.exists(path):
        print(f"[skip] {cond}: path not found ({path})")
        continue
 
    print(f"\nLoading {cond} from {path}")
    try:
        ft_body = load_transformer_body(path)
    except Exception as e:
        print(f"[ERROR] could not load {cond}: {e}")
        continue
 
    results[cond] = {}
    for layer in ALL_LAYERS:
        ft_w = get_block_weights(ft_body, layer)
        layer_res = {}
        for name in SUBMODULES:
            rc = rel_frobenius_change(base_weights[layer][name], ft_w[name])
            layer_res[name] = rc
        # aggregate: mean relative change across submodules for this layer
        layer_res["_mean"] = sum(layer_res[n] for n in SUBMODULES) / len(SUBMODULES)
        results[cond][layer] = layer_res
 
    del ft_body
 
 
# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
print("\n" + "=" * 78)
print("RELATIVE FROBENIUS WEIGHT CHANGE vs BASE GPT-2  (higher = more change)")
print("=" * 78)
 
for cond in results:
    print(f"\n### {cond}")
    print(f"{'layer':<8}" + "".join(f"{n:<16}" for n in SUBMODULES) + f"{'MEAN':<12}")
    for layer in ALL_LAYERS:
        r = results[cond][layer]
        row = f"{layer:<8}"
        row += "".join(f"{r[n]*100:<16.4f}" for n in SUBMODULES)
        row += f"{r['_mean']*100:<12.4f}"
        print(row + "  (%)")
 
# ----------------------------------------------------------------------
# Verdict on primary layer (the one used in SAE runs)
# ----------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"GO / NO-GO SUMMARY  (primary SAE layer = {PRIMARY_LAYER})")
print("=" * 78)
 
THRESHOLD_GO = 1.0       # >1% mean relative change on full_ft = clearly enough
THRESHOLD_MARGINAL = 0.1 # <0.1% = too mild, switch task
 
if "full_ft" in results:
    full_mean = results["full_ft"][PRIMARY_LAYER]["_mean"] * 100
    print(f"full_ft mean rel. change @ layer {PRIMARY_LAYER}: {full_mean:.4f}%")
    if full_mean >= THRESHOLD_GO:
        verdict = "GO -- fine-tuning clearly moved weights, proceed with SAE runs."
    elif full_mean >= THRESHOLD_MARGINAL:
        verdict = "MARGINAL -- some change but weak. Consider SST-2 for stronger signal."
    else:
        verdict = "NO-GO -- change too small. Switch to SST-2 or MNLI and re-fine-tune."
    print(verdict)
 
    # monotonicity check across LoRA ranks
    lora_conds = [c for c in ["lora_r4", "lora_r8", "lora_r16", "lora_r32"] if c in results]
    if lora_conds:
        print("\nLoRA rank monotonicity @ primary layer (expect increasing with rank):")
        for c in lora_conds:
            print(f"  {c:<10}: {results[c][PRIMARY_LAYER]['_mean']*100:.4f}%")
        vals = [results[c][PRIMARY_LAYER]["_mean"] for c in lora_conds]
        monotone = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        print(f"  monotone increasing: {monotone}")
else:
    print("full_ft results missing -- cannot render verdict.")
    verdict = "UNKNOWN"
 
# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
out = {
    "base_model": BASE_MODEL,
    "target_layers": ALL_LAYERS,
    "primary_layer": PRIMARY_LAYER,
    "submodules": SUBMODULES,
    "relative_change_percent": {
        cond: {
            str(layer): {n: results[cond][layer][n] * 100 for n in list(SUBMODULES) + ["_mean"]}
            for layer in ALL_LAYERS
        }
        for cond in results
    },
}
out_path = os.path.join(SAVE_DIR, "weight_change_diagnostic.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved results to {out_path}")
 
