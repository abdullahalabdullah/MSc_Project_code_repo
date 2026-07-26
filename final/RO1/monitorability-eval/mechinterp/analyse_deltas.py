"""Post-hoc analysis of the activation delta vectors from act_extract.py.

Separates the GENERIC persona shift from the HINT-SPECIFIC component:

  Delta_cued   = mean over hinted prompts of (act_adapter - act_base)
  Delta_uncued = same on unhinted prompts
  delta_hint   = Delta_cued - Delta_uncued   (the representational change caused
                 specifically by the hint's presence — most relevant to whether
                 the model will verbalize the hint)

Key questions:
  1. Is each adapter's shift generic?   cos(Delta_cued, Delta_uncued) per model.
  2. Do the two adapters converge more on the hint-specific part?
       cos(delta_H_hint, delta_S_hint)  vs  cos(Delta_H_cued, Delta_S_cued)

Run:  python monitorability-eval/mechinterp/analyse_deltas.py
"""
import os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NPZ = os.path.join(REPO, "outputs-and-results", "mechinterp", "act_full", "delta_vectors.npz")


def cos(a, b, eps=1e-12):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def main():
    d = np.load(NPZ)
    dHc, dHu = d["delta_H_hinted_last"], d["delta_H_unhinted_last"]
    dSc, dSu = d["delta_S_hinted_last"], d["delta_S_unhinted_last"]
    nL = dHc.shape[0]
    H = dHc.shape[1]
    floor2 = 2 / np.sqrt(H)

    print(f"hidden={H}  layers(+emb)={nL}  random-cos 2sigma~{floor2:.3f}\n")
    print("Per-model self-cosine (cued vs uncued) + hint-specific fraction of the shift")
    print(f"{'L':>3} {'H c~u':>7} {'S c~u':>7} {'|dH_c|':>8} {'|dH_hint|':>9} {'hint%':>6}")
    sel = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 35]
    for L in sel:
        dHh = dHc[L] - dHu[L]
        frac = np.linalg.norm(dHh) / (np.linalg.norm(dHc[L]) + 1e-12)
        print(f"{L:3d} {cos(dHc[L], dHu[L]):7.3f} {cos(dSc[L], dSu[L]):7.3f} "
              f"{np.linalg.norm(dHc[L]):8.3f} {np.linalg.norm(dHh):9.3f} {100*frac:6.1f}")

    print("\nHint-specific direction agreement vs raw cued agreement")
    print(f"{'L':>3} {'cos(dH_hint,dS_hint)':>21} {'cos(DH_cued,DS_cued)':>21}")
    cds, ccs = [], []
    for L in range(nL):
        dHh, dSh = dHc[L] - dHu[L], dSc[L] - dSu[L]
        cds.append(cos(dHh, dSh)); ccs.append(cos(dHc[L], dSc[L]))
    for L in sel:
        print(f"{L:3d} {cds[L]:21.3f} {ccs[L]:21.3f}")
    print(f"\nmean over layers 5-35:  hint-specific {np.mean(cds[5:36]):+.3f}   "
          f"raw cued {np.mean(ccs[5:36]):+.3f}   (2sigma~{floor2:.3f})")
    best = 5 + int(np.argmax(cds[5:36]))
    print(f"peak hint-specific agreement: {cds[best]:+.3f} @ L{best}")


if __name__ == "__main__":
    main()
