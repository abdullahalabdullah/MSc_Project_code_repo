"""Activation-space convergence harness for the two DPO interventions.

Question: do harmless-DPO and anti-syc-DPO shift the residual stream along a
COMMON direction, even though their LoRA *weights* are near-orthogonal
(see lora_diff.py: weight-space cos(H,S) ~ 0.05)?

Method (paired, adapter-swap on one base):
  Load Qwen3-4B once, attach both adapters via PEFT. For each cued (hinted)
  prompt p and layer L read the residual stream h_{m,L}(p) for
  m in {base B (disable_adapter), +harmless H, +anti-syc S} at two readouts:
     last : final prompt token (drives the first generated token)  <- headline
     mean : mean over prompt tokens (NOTE: dominated by Qwen massive-activation
            sink tokens at deep layers -> unreliable, kept only for reference)
  Per-prompt diff  d^H_L(p) = h_{H,L}(p) - h_{B,L}(p),  d^S similarly.
  Layer direction  Delta^H_L = mean_p d^H_L(p),  Delta^S_L likewise.
  Headline         cos(Delta^H_L, Delta^S_L)  per layer.

Also emits the uncued arm so downstream analysis can isolate the hint-specific
component  delta = Delta_cued - Delta_uncued  (see analyse_deltas.py).

Prompts are byte-identical to the RO1 monitorability run (same system prompt,
Tulu MCQ block, hint prefixes, Qwen3 thinking-mode template) via format_prompt_qwen.

Saves Delta vectors (per layer/readout/arm) to delta_vectors.npz — these are the
steering vectors consumed by cross_steer.py.

Run (full, ~28 min on M-series MPS):
  python monitorability-eval/mechinterp/act_extract.py --n-per-cell 12
"""
import argparse, json, os, sys, time
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "monitorability-eval"))
from src.inference_qwen import format_prompt_qwen  # noqa: E402

BASE = "Qwen/Qwen3-4B"
ADAPTERS = {
    "H": "er-abd/harmless-dpo-qwen3-4B-step-735",
    "S": "er-abd/anti-syc-dpo-qwen3-4B-step-520",
}
ALL_DATASETS = ["mmlu_pro", "gpqa_diamond", "wmdp", "codemmlu",
                "lsat_ar", "lsat_lr", "medxpertqa", "lexam"]
ALL_HINTS = ["professor", "grader_hacking", "unethical"]
DATA = os.path.join(REPO, "monitorability-eval", "data")
OUT_DEFAULT = os.path.join(REPO, "outputs-and-results", "mechinterp", "act_full")


def load_prompts(datasets, hints, n_per_cell, tokenizer, arms):
    rows = []
    for ds in datasets:
        idx = json.load(open(os.path.join(DATA, ds, "dataset_index.json")))
        for it in idx[:n_per_cell]:  # deterministic first-N of 200
            if "hinted" in arms:
                for ht in hints:
                    rows.append(dict(
                        prompt=format_prompt_qwen(it, True, tokenizer, hint_type=ht),
                        arm="hinted", dataset=ds, hint=ht, instance_id=it["instance_id"]))
            if "unhinted" in arms:
                rows.append(dict(
                    prompt=format_prompt_qwen(it, False, tokenizer, hint_type=hints[0]),
                    arm="unhinted", dataset=ds, hint="_none", instance_id=it["instance_id"]))
    return rows


@torch.no_grad()
def hidden_readouts(model, input_ids, attn):
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True,
                use_cache=False)
    hs = out.hidden_states  # tuple len L+1, each (1, seq, H)
    last = torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy()
    mean = torch.stack([h[0].mean(0) for h in hs]).float().cpu().numpy()
    del out
    return last, mean


def cos(a, b, eps=1e-12):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def subspace_overlap(D1, D2, k):
    """centered top-k PCA subspace overlap of per-prompt diffs. ~k/H if random."""
    def basis(D):
        U, S, Vt = np.linalg.svd(D - D.mean(0, keepdims=True), full_matrices=False)
        return Vt[:k].T
    Q1, Q2 = basis(D1), basis(D2)
    return float(np.sum((Q1.T @ Q2) ** 2) / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=12)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--hints", nargs="+", default=ALL_HINTS)
    ap.add_argument("--arms", nargs="+", default=["hinted", "unhinted"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pca-k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print("loading base + adapters...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype)
    model = PeftModel.from_pretrained(model, ADAPTERS["H"], adapter_name="H")
    model.load_adapter(ADAPTERS["S"], adapter_name="S")
    model.to(args.device).eval()
    n_hs = model.config.num_hidden_layers + 1
    H = model.config.hidden_size

    rows = load_prompts(args.datasets, args.hints, args.n_per_cell, tok, args.arms)
    print(f"{len(rows)} prompts | layers(+emb)={n_hs} hidden={H} device={args.device} "
          f"dtype={args.dtype}", flush=True)

    diffs = {arm: {ro: {"H": [], "S": []} for ro in ("last", "mean")} for arm in args.arms}
    t0 = time.time()
    for i, r in enumerate(rows):
        enc = tok(r["prompt"], return_tensors="pt", truncation=True, max_length=args.max_len)
        ids = enc.input_ids.to(args.device)
        am = enc.attention_mask.to(args.device)
        with model.disable_adapter():
            b_last, b_mean = hidden_readouts(model, ids, am)
        acts = {}
        for name in ("H", "S"):
            model.set_adapter(name)
            acts[name] = hidden_readouts(model, ids, am)
        for name in ("H", "S"):
            a_last, a_mean = acts[name]
            diffs[r["arm"]]["last"][name].append(a_last - b_last)
            diffs[r["arm"]]["mean"][name].append(a_mean - b_mean)
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            dt = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {dt:.0f}s  {dt/(i+1):.2f}s/prompt", flush=True)

    rng = np.random.default_rng(0)
    rfloor = np.array([cos(rng.standard_normal(H), rng.standard_normal(H)) for _ in range(2000)])
    floor_std = rfloor.std()
    report = {"config": vars(args), "hidden": H, "n_layers_plus_emb": n_hs,
              "random_floor_mean": float(rfloor.mean()), "random_floor_std": float(floor_std),
              "n_prompts": {arm: len(diffs[arm]["last"]["H"]) for arm in args.arms}}
    npz = {}
    for arm in args.arms:
        for ro in ("last", "mean"):
            DH = np.stack(diffs[arm][ro]["H"])
            DS = np.stack(diffs[arm][ro]["S"])
            deltaH, deltaS = DH.mean(0), DS.mean(0)
            npz[f"delta_H_{arm}_{ro}"] = deltaH
            npz[f"delta_S_{arm}_{ro}"] = deltaS
            report[f"{arm}_{ro}"] = [dict(
                layer=L, cos_HS=cos(deltaH[L], deltaS[L]),
                norm_H=float(np.linalg.norm(deltaH[L])),
                norm_S=float(np.linalg.norm(deltaS[L])),
                subspace_HS=subspace_overlap(DH[:, L, :], DS[:, L, :], args.pca_k),
            ) for L in range(n_hs)]

    json.dump(report, open(os.path.join(args.out, "act_report.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(args.out, "delta_vectors.npz"), **npz)

    def band(x):
        return "~floor" if abs(x) < 2 * floor_std else ("*" if abs(x) < 5 * floor_std else "**")
    print(f"\nrandom floor: mean {rfloor.mean():+.4f} std {floor_std:.4f} "
          f"(2s={2*floor_std:.3f}, 5s={5*floor_std:.3f})")
    for arm in args.arms:
        print(f"\n=== cos(Delta_H, Delta_S) per layer — arm={arm} ===")
        print(f"{'L':>3} {'last':>8} {'':3} {'mean':>8} {'':3} {'sub_last':>8} {'|dH|':>8} {'|dS|':>8}")
        for L in range(n_hs):
            a = report[f"{arm}_last"][L]; b = report[f"{arm}_mean"][L]
            print(f"{L:3d} {a['cos_HS']:+8.4f} {band(a['cos_HS']):>3} "
                  f"{b['cos_HS']:+8.4f} {band(b['cos_HS']):>3} "
                  f"{a['subspace_HS']:8.4f} {a['norm_H']:8.3f} {a['norm_S']:8.3f}")
        cl = [report[f'{arm}_last'][L]['cos_HS'] for L in range(n_hs)]
        print(f"  mean cos over layers (last): {np.mean(cl):+.4f}  "
              f"max {np.max(cl):+.4f} @L{int(np.argmax(cl))}")
    print(f"\nsaved -> {args.out}/act_report.json , delta_vectors.npz")


if __name__ == "__main__":
    main()
