"""Weight-space convergence analysis of the DPO/SFT LoRA adapters.

For each adapter, DeltaW = (alpha/r) * B @ A per target module IS the
(finetuned - base) weight diff. We compare adapters pairwise WITHOUT ever
materializing DeltaW (each full matrix is ~14 GB); all math is done in the
r x r / d_out x r factor space:
  flat cosine   <B1A1, B2A2>_F = tr((B2^T B1) @ (A1 @ A2^T))
  ||BA||_F^2    = tr((B^T B) @ (A A^T))
  subspace overlap: col(BA) = col(B); orthonormalize B (QR) and compute
    ||Q1^T Q2||_F^2 / r  (~r/d for random r-dim subspaces, 1 if identical).

Pairs: (H,S) crux; (H,SFT),(S,SFT) shared-data/artifact floor; (H147,H735) dose.

Run:  python monitorability-eval/mechinterp/lora_diff.py
Out:  outputs-and-results/mechinterp/lora_diff_results.json  + stdout tables
"""
import json, os
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

torch.set_grad_enabled(False)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "outputs-and-results", "mechinterp")

ADAPTERS = {
    "H735": "er-abd/harmless-dpo-qwen3-4B-step-735",
    "H147": "er-abd/harmless-dpo-qwen3-4B-step-147",
    "S520": "er-abd/anti-syc-dpo-qwen3-4B-step-520",
    "SFT": "er-abd/harmless-sft-qwen3-4B-step-735",
}


def load_adapter(repo):
    cfg = json.load(open(hf_hub_download(repo, "adapter_config.json")))
    weights = load_file(hf_hub_download(repo, "adapter_model.safetensors"))
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    mods = defaultdict(dict)
    for k, v in weights.items():
        if ".lora_A." in k:
            mods[k.replace(".lora_A.weight", "")]["A"] = v.float()
        elif ".lora_B." in k:
            mods[k.replace(".lora_B.weight", "")]["B"] = v.float()
    out = {name: (ab["A"], ab["B"], scale) for name, ab in mods.items()
           if "A" in ab and "B" in ab}
    return cfg, out


def inner(m1, m2):
    A1, B1, s1 = m1
    A2, B2, s2 = m2
    return s1 * s2 * torch.trace((B2.T @ B1) @ (A1 @ A2.T)).item()


def fro2(m):
    A, B, s = m
    return s * s * torch.trace((B.T @ B) @ (A @ A.T)).item()


def flat_cos(m1, m2):
    return inner(m1, m2) / ((fro2(m1) * fro2(m2)) ** 0.5 + 1e-12)


def b_basis(m):
    return torch.linalg.qr(m[1]).Q  # d_out x r orthonormal


def short(name):
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1]), parts[-1]


def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading adapters...", flush=True)
    data = {}
    for tag, repo in ADAPTERS.items():
        cfg, mods = load_adapter(repo)
        data[tag] = mods
        print(f"  {tag}: r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora')} "
              f"modules={len(mods)}", flush=True)
    R = cfg["r"]

    PAIRS = [("H735", "S520"), ("H735", "SFT"), ("S520", "SFT"), ("H147", "H735")]
    common = sorted(set.intersection(*[set(d.keys()) for d in data.values()]), key=short)
    print(f"common modules: {len(common)}", flush=True)

    results = {}
    for name in common:
        layer, mod = short(name)
        Qs = {t: b_basis(data[t][name]) for t in ADAPTERS}
        entry = {"d_out": Qs["H735"].shape[0]}
        for t1, t2 in PAIRS:
            c = flat_cos(data[t1][name], data[t2][name])
            ov = ((Qs[t1].T @ Qs[t2]) ** 2).sum().item() / R
            entry[(t1, t2)] = (c, ov)
        results[(layer, mod)] = entry

    mods = sorted({m for (_, m) in results})
    layers = sorted({l for (l, _) in results})

    def agg(pair, sel=None):
        cs, ovs, floor = [], [], []
        for key, d in results.items():
            if sel and not sel(*key):
                continue
            c, ov = d[pair]
            cs.append(c); ovs.append(ov); floor.append(R / d["d_out"])
        n = len(cs)
        return sum(cs) / n, sum(ovs) / n, sum(floor) / n

    print("\n=== Overall (mean across all 252 modules) ===")
    print(f"{'pair':14} {'flat_cos':>9} {'subspace_overlap':>17} {'random_floor':>13}")
    for pair in PAIRS:
        c, ov, fl = agg(pair)
        print(f"{pair[0] + '~' + pair[1]:14} {c:9.4f} {ov:17.4f} {fl:13.4f}")

    print("\n=== Per module type (mean across layers) ===")
    print(f"{'module':10} {'cos(H,S)':>9} {'ov(H,S)':>8} {'cos(H,SFT)':>11} {'ov(H,SFT)':>10} "
          f"{'cos(S,SFT)':>11} {'ov(S,SFT)':>10} {'floor':>7}")
    for m in mods:
        row = []
        for pair in [("H735", "S520"), ("H735", "SFT"), ("S520", "SFT")]:
            c, ov, fl = agg(pair, sel=lambda l, mm: mm == m)
            row += [c, ov]
        print(f"{m:10} {row[0]:9.4f} {row[1]:8.4f} {row[2]:11.4f} {row[3]:10.4f} "
              f"{row[4]:11.4f} {row[5]:10.4f} {fl:7.4f}")

    print("\n=== Per layer (mean across module types) ===")
    print(f"{'layer':>5} {'cos(H,S)':>9} {'ov(H,S)':>8} {'ov(H,SFT)':>10} {'ov(S,SFT)':>10} "
          f"{'cos(H147,H735)':>15} {'ov(H147,H735)':>14}")
    for l in layers:
        vals = [agg(p, sel=lambda ll, mm: ll == l) for p in PAIRS]
        print(f"{l:5d} {vals[0][0]:9.4f} {vals[0][1]:8.4f} {vals[1][1]:10.4f} {vals[2][1]:10.4f} "
              f"{vals[3][0]:15.4f} {vals[3][1]:14.4f}")

    print("\n=== DeltaW Frobenius norm totals ===")
    for tag in ADAPTERS:
        tot = sum(fro2(m) for m in data[tag].values()) ** 0.5
        print(f"  {tag}: {tot:.3f}")

    out = {f"{a}~{b}": {f"{l}.{m}": results[(l, m)][(a, b)] for (l, m) in results}
           for a, b in PAIRS}
    json.dump(out, open(os.path.join(OUT, "lora_diff_results.json"), "w"))
    print(f"\nsaved -> {OUT}/lora_diff_results.json")


if __name__ == "__main__":
    main()
