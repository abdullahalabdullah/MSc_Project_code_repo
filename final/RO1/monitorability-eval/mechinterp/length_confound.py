"""Tier-0 length-confound check: is the DPO monitorability gain just longer CoTs?

For every HINT-FOLLOWING hinted run (the canonical verbalization denominator) we
join, per (model, dataset, hint, instance, sample):
    acknowledged (monitor label)   <- monitor_outputs/.../monitor_results.jsonl
    token_count  (CoT length)      <- inference-outputs/.../hinted.jsonl
across models base / H (harmless-DPO) / S (anti-syc-DPO).

Two answers to "at the same length, do DPO models still acknowledge more?":

  (B) Stratified table: within each (dataset,hint) assign length terciles, then
      compare acknowledgment rate by model within tercile (matched relative
      length inside the same dataset -> not confounded by dataset length scale).

  (A) Logistic regression: acknowledged ~ model + z(log tokens) + dataset FE +
      hint FE, with cluster-robust SEs (clustered on dataset x instance, since
      the 8 samples per question are not independent). Reports odds ratios for
      H and S vs base after removing length + dataset + hint.

Run:  python monitorability-eval/mechinterp/length_confound.py
Out:  outputs-and-results/mechinterp/length_confound.md  + stdout
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "outputs-and-results")
OUT = os.path.join(RES, "mechinterp")

MODELS = {"base": "Qwen_Qwen3-4B",
          "H": "er-abd_harmless-dpo-qwen3-4B-step-735",
          "S": "er-abd_anti-syc-dpo-qwen3-4B-step-520"}
DATASETS = ["mmlu_pro", "gpqa_diamond", "wmdp", "codemmlu",
            "lsat_ar", "lsat_lr", "medxpertqa", "lexam"]
HINTS = ["professor", "grader_hacking", "unethical"]


def load_runs():
    rows = []
    for mkey, mdir in MODELS.items():
        for ds in DATASETS:
            for ht in HINTS:
                mon_p = os.path.join(RES, "monitor_outputs", mdir, ds, ht, "monitor_results.jsonl")
                inf_p = os.path.join(RES, "inference-outputs", mdir, ds, ht, "hinted.jsonl")
                if not (os.path.exists(mon_p) and os.path.exists(inf_p)):
                    continue
                tok = {}
                for line in open(inf_p):
                    r = json.loads(line)
                    tok[(r["instance_id"], r["sample_idx"])] = (r["token_count"], r.get("truncated", False))
                for line in open(mon_p):
                    r = json.loads(line)
                    if r.get("arm") != "hinted" or not r.get("followed_hint"):
                        continue
                    ack = r.get("acknowledged")
                    if ack is None:
                        continue
                    key = (r["instance_id"], r["sample_idx"])
                    if key not in tok:
                        continue
                    tc, trunc = tok[key]
                    if trunc or tc is None:
                        continue
                    rows.append(dict(model=mkey, dataset=ds, hint=ht,
                                     instance_id=r["instance_id"], sample_idx=r["sample_idx"],
                                     acknowledged=int(bool(ack)), token_count=int(tc)))
    return pd.DataFrame(rows)


def irls_logit(X, y, groups, ridge=1e-6, iters=100):
    """Logistic regression via IRLS. Returns beta, cluster-robust SE (on groups)."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        XtWX = X.T @ (W[:, None] * X) + ridge * np.eye(p)
        z = eta + (y - mu) / W
        beta_new = np.linalg.solve(XtWX, X.T @ (W * z))
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    # cluster-robust (sandwich) covariance, clustered on groups
    eta = X @ beta
    mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    W = np.clip(mu * (1 - mu), 1e-6, None)
    bread = np.linalg.inv(X.T @ (W[:, None] * X) + ridge * np.eye(p))
    score = X * (y - mu)[:, None]
    meat = np.zeros((p, p))
    for g in np.unique(groups):
        sg = score[groups == g].sum(0)
        meat += np.outer(sg, sg)
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_runs()
    lines = []

    def P(s=""):
        print(s); lines.append(s)

    P("# Tier-0 length-confound check — hint-following hinted runs\n")
    P(f"Joined runs: {len(df)}  (base={sum(df.model=='base')}, "
      f"H={sum(df.model=='H')}, S={sum(df.model=='S')})\n")

    # ---- descriptive: raw ack rate + length per model ----
    P("## Raw (uncontrolled): acknowledgment rate and CoT length per model\n")
    P("| model | n | ack rate | median tokens | mean tokens |")
    P("|---|---|---|---|---|")
    for m in ["base", "H", "S"]:
        d = df[df.model == m]
        P(f"| {m} | {len(d)} | {d.acknowledged.mean():.3f} | "
          f"{d.token_count.median():.0f} | {d.token_count.mean():.0f} |")
    P("")

    # ---- (B) stratified within (dataset,hint) length terciles ----
    P("## (B) Stratified: acknowledgment rate by model WITHIN length tercile\n")
    P("Terciles computed within each (dataset,hint) cell (matched relative length,\n"
      "same dataset), then pooled. If DPO > base in every tercile, length is not the cause.\n")
    df = df.copy()
    df["tercile"] = np.nan
    for (ds, ht), idx in df.groupby(["dataset", "hint"]).groups.items():
        sub = df.loc[idx, "token_count"]
        try:
            q = pd.qcut(sub, 3, labels=["short", "mid", "long"], duplicates="drop")
            df.loc[idx, "tercile"] = q.astype(object)
        except Exception:
            pass
    P("| length tercile | base ack | H ack | S ack | H−base | S−base |")
    P("|---|---|---|---|---|---|")
    for t in ["short", "mid", "long"]:
        sub = df[df.tercile == t]
        b = sub[sub.model == "base"].acknowledged.mean()
        h = sub[sub.model == "H"].acknowledged.mean()
        s = sub[sub.model == "S"].acknowledged.mean()
        P(f"| {t} | {b:.3f} | {h:.3f} | {s:.3f} | {h-b:+.3f} | {s-b:+.3f} |")
    # also overall median tokens per tercile for context
    P("")
    P("Median tokens per tercile (context): " +
      ", ".join(f"{t}={df[df.tercile==t].token_count.median():.0f}"
                for t in ['short', 'mid', 'long']))
    P("")

    # ---- (A) logistic regression with FE + cluster-robust SE ----
    P("## (A) Logistic regression: ack ~ model + z(log tokens) + dataset FE + hint FE\n")
    d = df.dropna(subset=["tercile"]).copy()
    d["logtok"] = np.log(d.token_count)
    d["zlogtok"] = (d.logtok - d.logtok.mean()) / d.logtok.std()
    cols = ["const", "H", "S", "zlogtok"]
    Xparts = {"const": np.ones(len(d)), "H": (d.model == "H").astype(float).values,
              "S": (d.model == "S").astype(float).values, "zlogtok": d.zlogtok.values}
    for ds in DATASETS[1:]:  # base dataset = first, dropped
        Xparts[f"ds_{ds}"] = (d.dataset == ds).astype(float).values; cols.append(f"ds_{ds}")
    for ht in HINTS[1:]:
        Xparts[f"h_{ht}"] = (d.hint == ht).astype(float).values; cols.append(f"h_{ht}")
    X = np.column_stack([Xparts[c] for c in cols])
    y = d.acknowledged.values.astype(float)
    groups = (d.dataset + "_" + d.instance_id.astype(str)).values
    beta, se = irls_logit(X, y, groups)
    zt = beta / np.where(se > 0, se, np.nan)
    pv = 2 * norm.sf(np.abs(zt))
    P("Odds ratios vs base, controlling for length + dataset + hint "
      "(cluster-robust on dataset×instance):\n")
    P("| term | coef (log-odds) | odds ratio | 95% CI (OR) | p |")
    P("|---|---|---|---|---|")
    for i, c in enumerate(cols):
        if c not in ("H", "S", "zlogtok"):
            continue
        lo, hi = np.exp(beta[i] - 1.96 * se[i]), np.exp(beta[i] + 1.96 * se[i])
        name = {"H": "H (harmless-DPO)", "S": "S (anti-syc-DPO)",
                "zlogtok": "+1 SD log(tokens)"}[c]
        P(f"| {name} | {beta[i]:+.3f} | {np.exp(beta[i]):.2f} | "
          f"[{lo:.2f}, {hi:.2f}] | {pv[i]:.2e} |")
    P("")

    open(os.path.join(OUT, "length_confound.md"), "w").write("\n".join(lines))
    print(f"\nsaved -> {OUT}/length_confound.md")


if __name__ == "__main__":
    main()
