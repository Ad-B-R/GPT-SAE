import os

os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

import json
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

SAE_DIR = "./pretrained_saes"
LAYERS = list(range(12))
SAVE_DIR = "./diagnostics"
os.makedirs(SAVE_DIR, exist_ok=True)

N_PROBE_EXAMPLES = 500          # size of fixed probe corpus
BATCH_SIZE = 32
CONTEXT_LEN = 128                # premise+hypothesis pairs run longer than a single sentence


print("Loading base GPT-2 (HookedTransformer)...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

print("Loading probe corpus (MNLI)...")
# validation split, not train — this is an inference-only diagnostic, so it
# should stay independent of whatever data the fine-tuning scripts train on
raw = load_dataset("glue", "mnli", split="validation_matched")
texts = [
    f"{p.strip()} {h.strip()}"
    for p, h in zip(raw["premise"], raw["hypothesis"])
    if p.strip() and h.strip()
][:N_PROBE_EXAMPLES]
print(f"Using {len(texts)} probe examples.")

def load_sae_from_local(layer: int):
    path = os.path.join(SAE_DIR, f"layer_{layer}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No pretrained SAE found at {path}. "
            f"Run the loading/saving script first."
        )
    sae = SAE.load_from_disk(path=path, device=device)
    sae.eval()
    return sae


print("Loading pretrained SAEs from disk...")
saes = {layer: load_sae_from_local(layer) for layer in tqdm(LAYERS)}


def sae_reconstruct(sae, acts_flat: torch.Tensor) -> torch.Tensor:
    """
    Returns the reconstructed activations for a batch of flattened
    activations, robust to different SAELens return signatures.
    """
    with torch.no_grad():
        out = sae(acts_flat)

    if isinstance(out, torch.Tensor):
        return out
    # some SAELens versions return a dataclass / namedtuple with .sae_out
    if hasattr(out, "sae_out"):
        return out.sae_out
    # fallback: manual encode -> decode
    feats = sae.encode(acts_flat)
    return sae.decode(feats)

def compute_layer_reconstruction_stats(model, saes, texts, device,
                                        batch_size=BATCH_SIZE):
    """
    For every layer, compute:
      - mean MSE reconstruction loss
      - normalized MSE (MSE / variance of activations)   -> lower is better
      - fraction of variance explained (1 - normalized MSE) -> higher is better
    """
    layers = sorted(saes.keys())
    sq_err_sum = {l: 0.0 for l in layers}
    var_sum = {l: 0.0 for l in layers}
    n_tokens = {l: 0 for l in layers}

    hook_names = [f"blocks.{l}.hook_resid_pre" for l in layers]

    for i in tqdm(range(0, len(texts), batch_size), desc="Computing recon loss"):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        if tokens.shape[1] > CONTEXT_LEN:
            tokens = tokens[:, :CONTEXT_LEN]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in hook_names,
            )

        for l in layers:
            hook_name = f"blocks.{l}.hook_resid_pre"
            acts = cache[hook_name]                      # (batch, seq, d_model)
            acts_flat = acts.reshape(-1, acts.shape[-1])  # (batch*seq, d_model)

            recon = sae_reconstruct(saes[l], acts_flat)

            sq_err = (recon - acts_flat).pow(2).sum().item()
            var = acts_flat.var(dim=0, unbiased=False).sum().item() * acts_flat.shape[0]

            sq_err_sum[l] += sq_err
            var_sum[l] += var
            n_tokens[l] += acts_flat.shape[0]

        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

    results = {}
    for l in layers:
        mse = sq_err_sum[l] / (n_tokens[l] * saes[l].cfg.d_in)
        normalized_mse = sq_err_sum[l] / max(var_sum[l], 1e-8)
        frac_var_explained = 1 - normalized_mse
        results[l] = {
            "mse": mse,
            "normalized_mse": normalized_mse,
            "frac_var_explained": frac_var_explained,
        }
    return results


print("Running reconstruction diagnostic across all layers...")
results = compute_layer_reconstruction_stats(model, saes, texts, device)

print("\n" + "=" * 60)
print(f"{'Layer':<8}{'MSE':<15}{'Norm. MSE':<15}{'Frac Var Explained':<20}")
print("=" * 60)
for l in sorted(results.keys()):
    r = results[l]
    print(f"{l:<8}{r['mse']:<15.5f}{r['normalized_mse']:<15.5f}{r['frac_var_explained']:<20.5f}")

# rank layers by fraction of variance explained (higher = more structured)
ranked = sorted(results.items(), key=lambda kv: -kv[1]["frac_var_explained"])
print("\nLayers ranked by reconstruction quality (best first):")
for l, r in ranked:
    print(f"  Layer {l:2d} | frac_var_explained = {r['frac_var_explained']:.4f}")

# rank layers by raw reconstruction loss (lower = better)
ranked_by_mse = sorted(results.items(), key=lambda kv: kv[1]["mse"])
print("\nLayers ranked by reconstruction loss / MSE (best first):")
for l, r in ranked_by_mse:
    print(f"  Layer {l:2d} | mse = {r['mse']:.5f}")

top3 = [l for l, _ in ranked[:3]]
print(f"\nSuggested top-3 layers for full experiment: {sorted(top3)}")

with open(os.path.join(SAVE_DIR, "layer_reconstruction_diagnostic.json"), "w") as f:
    json.dump(
        {
            "results": results,
            "ranked_layers": [l for l, _ in ranked],
            "ranked_layers_by_mse": [l for l, _ in ranked_by_mse],
            "suggested_top3": sorted(top3),
        },
        f,
        indent=2,
    )
print(f"\nSaved results to {SAVE_DIR}/layer_reconstruction_diagnostic.json")

try:
    import matplotlib.pyplot as plt

    layers_sorted = sorted(results.keys())
    fve = [results[l]["frac_var_explained"] for l in layers_sorted]

    plt.figure(figsize=(8, 4))
    plt.bar(layers_sorted, fve, color="steelblue")
    plt.xlabel("Layer")
    plt.ylabel("Fraction of variance explained")
    plt.title("Base GPT-2 SAE reconstruction quality by layer")
    plt.xticks(layers_sorted)
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "layer_reconstruction_diagnostic.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
except ImportError:
    print("matplotlib not installed, skipping plot.")