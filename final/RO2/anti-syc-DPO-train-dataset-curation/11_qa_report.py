"""Stage J (11) — QA report, manifest, pilot gates (CPU).

Writes everything spec §11 requires:
  manifest.json    — full §2 config snapshot, all seeds, git hash, timestamps, counts
                     at every stage boundary, classifier call volumes, sha256 of the
                     frozen reserved/ files, the gpt-5-mini temperature-omitted note.
  qa_report.md     — funnel, Filter-0 consistency distribution, per-stratum Filter-1
                     survival, syc_fraction distribution (histogram PNG + mean/median,
                     overall & per template), per-template survival/class rates,
                     grader-audit results (or PENDING), UNPARSEABLE rate + truncation
                     vs missing-line attribution, direction composition, main/sham
                     counts, chosen-vs-rejected token lengths (>20% mean-gap flag).
  pilot_gates.json — only with PILOT=1: the four blocking gates of spec §10.

Always safe to re-run (pure reporting; no caching marker — it re-reads whatever exists).
"""
import argparse
import datetime
import json
import os
import statistics
import subprocess

import config
import env_setup
import prompts
from utils import read_jsonl, sha256_file

STAGE = "11_qa_report"


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.HERE,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _mean_median(values):
    if not values:
        return None, None
    return round(statistics.mean(values), 1), round(statistics.median(values), 1)


def _stage_counts():
    counts = {}
    for marker in sorted(os.listdir(config.MARKERS_DIR)) if os.path.isdir(config.MARKERS_DIR) else []:
        try:
            with open(os.path.join(config.MARKERS_DIR, marker)) as f:
                m = json.load(f)
            counts[m["stage"]] = {"counts": m.get("counts", {}),
                                  "completed_at": m.get("completed_at")}
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return counts


def _unparseable_stats(grades_paths):
    total = unparseable = trunc = no_line = invalid = api_calls = cached = 0
    for path in grades_paths:
        if not os.path.exists(path):
            continue
        for r in read_jsonl(path):
            total += 1
            if r.get("cached"):
                cached += 1
            elif r["label"] != "UNPARSEABLE" or r.get("unparseable_reason") == "classifier_output_invalid":
                api_calls += 1
            if r["label"] == "UNPARSEABLE":
                unparseable += 1
                reason = r.get("unparseable_reason")
                if reason == "truncated":
                    trunc += 1
                elif reason == "no_final_answer_line":
                    no_line += 1
                else:
                    invalid += 1
    return {
        "total_samples": total,
        "unparseable": unparseable,
        "unparseable_rate": round(unparseable / max(total, 1), 4),
        "from_truncation": trunc,
        "from_missing_final_answer_line": no_line,
        "from_invalid_classifier_output": invalid,
        "classifier_api_calls": api_calls,
        "classifier_cache_hits": cached,
    }


def _syc_histogram(values, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values, bins=20, range=(0, 1), edgecolor="black")
        ax.set_xlabel("syc_fraction (MATCHES_W / classified samples)")
        ax.set_ylabel("wrong-opinion instances")
        ax.set_title("Baseline sycophancy distribution (all wrong-opinion instances)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"WARNING: could not write histogram ({e})")
        return False


def main():
    ap = argparse.ArgumentParser(description="Stage J — QA report + manifest (+ pilot gates).")
    ap.parse_args()

    env_setup.setup_env()
    config.ensure_output_dirs()

    f0 = read_jsonl(config.FILTER0_SUMMARY_PATH) if os.path.exists(config.FILTER0_SUMMARY_PATH) else []
    f1 = read_jsonl(config.FILTER1_SUMMARY_PATH) if os.path.exists(config.FILTER1_SUMMARY_PATH) else []
    pairs = read_jsonl(config.PAIRS_PATH) if os.path.exists(config.PAIRS_PATH) else []
    sham = read_jsonl(config.SHAM_PAIRS_PATH) if os.path.exists(config.SHAM_PAIRS_PATH) else []
    pool = read_jsonl(config.TRAINING_POOL_PATH) if os.path.exists(config.TRAINING_POOL_PATH) else []
    instances = read_jsonl(config.INSTANCES_PATH) if os.path.exists(config.INSTANCES_PATH) else []

    # ── Funnel ───────────────────────────────────────────────────────────────
    n_f0_pass = sum(1 for s in f0 if s["pass"])
    n_f1_pass = sum(1 for s in f1 if s["pass"])
    funnel = {
        "sampled_pool": len(pool),
        "filter0_graded": len(f0),
        "filter0_pass": n_f0_pass,
        "instances": len(instances),
        "filter1_pass": n_f1_pass,
        "pairs_selected": n_f1_pass,
        "pairs_post_balance": len(pairs),
        "sham_pairs": len(sham),
    }

    # ── Filter-0 consistency distribution + per-stratum Filter-1 survival ────
    strata_order = ["70-80", "80-90", "90-99", "100"]
    consistency_dist = {s: 0 for s in strata_order}
    consistency_dist["below_70 (fail)"] = 0
    stratum_of = {}
    for s in f0:
        st = s.get("consistency_stratum")
        if st:
            consistency_dist[st] += 1
            stratum_of[s["question_id"]] = st
        else:
            consistency_dist["below_70 (fail)"] += 1
    stratum_survival = {}
    for s in f1:
        st = stratum_of.get(s["question_id"])
        if not st:
            continue
        d = stratum_survival.setdefault(st, {"instances": 0, "pass": 0})
        d["instances"] += 1
        d["pass"] += s["pass"]

    # ── syc_fraction distribution (ALL wrong-opinion instances, spec §7) ─────
    wrong = [s for s in f1 if s["direction"] == "wrong_opinion"]
    syc_values = [s["syc_fraction"] for s in wrong if s["syc_fraction"] is not None]
    syc_mean, syc_median = _mean_median(syc_values)
    have_hist = _syc_histogram(syc_values, config.SYC_FRACTION_FIG_PATH) if syc_values else False

    # ── Per-template stats ────────────────────────────────────────────────────
    per_template = {}
    for s in f1:
        t = per_template.setdefault(s["template_id"], {
            "instances": 0, "pass": 0,
            "wrong_instances": 0, "wrong_pass": 0,
            "wrong_classified_samples": 0, "wrong_rejected_samples": 0,
            "syc_values": [],
        })
        t["instances"] += 1
        t["pass"] += s["pass"]
        if s["direction"] == "wrong_opinion":
            t["wrong_instances"] += 1
            t["wrong_pass"] += s["pass"]
            t["wrong_classified_samples"] += s["n_classified"]
            t["wrong_rejected_samples"] += s["n_rejected_class"]
            if s["syc_fraction"] is not None:
                t["syc_values"].append(s["syc_fraction"])
    template_rows = {}
    for tid in sorted(per_template):
        t = per_template[tid]
        mean_syc, _ = _mean_median(t["syc_values"])
        template_rows[tid] = {
            "instances": t["instances"],
            "survival_rate": round(t["pass"] / max(t["instances"], 1), 4),
            # gate-3 quantity: rejected-class samples / classified samples,
            # wrong-opinion instances only
            "wrong_rejected_class_rate": round(
                t["wrong_rejected_samples"] / max(t["wrong_classified_samples"], 1), 4),
            "wrong_mean_syc_fraction": mean_syc,
        }

    # ── UNPARSEABLE + call volume ─────────────────────────────────────────────
    unparse = _unparseable_stats([config.FILTER0_GRADES_PATH, config.FILTER1_GRADES_PATH])
    n_wrong_answer_calls = 0
    if os.path.exists(config.WRONG_ANSWER_CACHE_PATH):
        n_wrong_answer_calls = len(read_jsonl(config.WRONG_ANSWER_CACHE_PATH))

    # ── Direction composition + length stats over the FINAL file ─────────────
    n_wrong_final = sum(1 for p in pairs if p["direction"] == "wrong_opinion")
    chosen_tok = [p["chosen"]["num_tokens"] for p in pairs]
    rejected_tok = [p["rejected"]["num_tokens"] for p in pairs]
    c_mean, c_median = _mean_median(chosen_tok)
    r_mean, r_median = _mean_median(rejected_tok)
    length_flag = (c_mean and r_mean
                   and abs(c_mean - r_mean) / max(min(c_mean, r_mean), 1) > 0.20)

    # ── Grader audit (may be pending) ─────────────────────────────────────────
    audit = None
    if os.path.exists(config.AUDIT_RESULTS_PATH):
        with open(config.AUDIT_RESULTS_PATH) as f:
            audit = json.load(f)

    # ── manifest.json ─────────────────────────────────────────────────────────
    manifest = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
        "config": {
            "x": config.X,
            "x_full": config.X_FULL,
            "pilot": config.PILOT,
            "eval_slice_size": config.EVAL_SLICE_SIZE,
            "n_filter0": config.N,
            "n_filter1": config.N_FILTER1,
            "sampling": config.sampling_config(),
            "filter0_pass_band": {"min_parseable": config.FILTER0_MIN_PARSEABLE,
                                  "min_consistency": config.FILTER0_MIN_CONSISTENCY},
            "grader_model": config.GRADER_MODEL,
            "grader_temperature_note": (
                "spec says temperature=0 but gpt-5-mini rejects non-default temperature; "
                "it is omitted from every call (API default), matching the existing "
                "monitorability monitor"),
            "dropped_templates": config.DROPPED_TEMPLATES,
            "valid_fraction": config.VALID_FRACTION,
            "dataset": {"name": config.TRIVIAQA_DATASET, "config": config.TRIVIAQA_CONFIG,
                        "split": config.TRIVIAQA_SPLIT},
            "ood_eval_url": config.OOD_EVAL_URL,
        },
        "seeds": {
            "sampling": config.SEED_SAMPLING, "template": config.SEED_TEMPLATE,
            "pair": config.SEED_PAIR, "sham": config.SEED_SHAM,
            "balance": config.SEED_BALANCE, "gen_seed_vllm": config.GEN_SEED,
            "split_seed_export": config.SPLIT_SEED,
        },
        "funnel": funnel,
        "stage_markers": _stage_counts(),
        "call_volume": {
            "classifier": {k: unparse[k] for k in
                           ("classifier_api_calls", "classifier_cache_hits")},
            "wrong_answer_generator_unique_calls": n_wrong_answer_calls,
        },
        "reserved_sha256": {
            os.path.basename(p): (sha256_file(p) if os.path.exists(p) else None)
            for p in (config.EVAL_ID_PATH, config.EVAL_OOD_PATH)
        },
    }
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest -> {config.MANIFEST_PATH}")

    # ── pilot_gates.json (PILOT only) ────────────────────────────────────────
    gates = None
    if config.PILOT:
        projected = (len(pairs) / max(config.X, 1)) * config.X_FULL
        drop_list = [tid for tid, row in template_rows.items()
                     if row["wrong_rejected_class_rate"] < 0.02]
        gates = {
            "gate1_grader_audit": {
                "status": ("PASS" if audit and audit.get("agreement", 0) >= 0.95
                           else "FAIL" if audit else "PENDING"),
                "agreement": audit.get("agreement") if audit else None,
                "required": 0.95,
                "note": "run audit_grader_sample.py --make / --score",
            },
            "gate2_yield_projection": {
                "status": "PASS" if projected >= config.MIN_PAIRS_FULL_RUN else "FAIL",
                "pilot_pairs_post_balance": len(pairs),
                "pilot_x": config.X,
                "x_full": config.X_FULL,
                "projected_pairs": round(projected),
                "required": config.MIN_PAIRS_FULL_RUN,
                "note_on_fail": "escalation path: N_FILTER1=16 (owner decision)",
            },
            "gate3_template_health": {
                "status": "PASS" if not drop_list else "DROP_TEMPLATES",
                "threshold_wrong_rejected_class_rate": 0.02,
                "per_template": {t: r["wrong_rejected_class_rate"]
                                 for t, r in template_rows.items()},
                "templates_to_drop": drop_list,
                "note": ("set DROPPED_TEMPLATES=" + ",".join(drop_list) +
                         " for the full run") if drop_list else None,
            },
            "gate4_parse_health": {
                "status": "PASS" if unparse["unparseable_rate"] < 0.10 else "FAIL",
                "unparseable_rate": unparse["unparseable_rate"],
                "required": "< 0.10",
            },
        }
        with open(config.PILOT_GATES_PATH, "w") as f:
            json.dump(gates, f, indent=2, ensure_ascii=False)
        print(f"Pilot gates -> {config.PILOT_GATES_PATH}")
        for name, g in gates.items():
            print(f"  {name}: {g['status']}")

    # ── qa_report.md ──────────────────────────────────────────────────────────
    lines = []
    a = lines.append
    a("# Anti-Sycophancy DPO Curation — QA Report")
    a("")
    a(f"Generated: {manifest['created_at']}  |  git: `{manifest['git_hash']}`  |  "
      f"X={config.X}  N={config.N}/{config.N_FILTER1}  PILOT={config.PILOT}")
    a("")
    a("## Funnel")
    a("")
    a("| Stage boundary | Count |")
    a("|---|---|")
    for k, v in funnel.items():
        a(f"| {k} | {v} |")
    a("")
    a("## Filter-0 consistency distribution")
    a("")
    a("| Stratum | Questions |")
    a("|---|---|")
    for k, v in consistency_dist.items():
        a(f"| {k} | {v} |")
    a("")
    a("### Per-stratum Filter-1 survival")
    a("")
    a("| Stratum | Instances | Pass | Survival |")
    a("|---|---|---|---|")
    for st in strata_order:
        d = stratum_survival.get(st)
        if d:
            a(f"| {st} | {d['instances']} | {d['pass']} | "
              f"{d['pass'] / max(d['instances'], 1):.1%} |")
    a("")
    a("## Baseline sycophancy (syc_fraction over ALL wrong-opinion instances)")
    a("")
    a(f"- instances with a defined syc_fraction: {len(syc_values)} / {len(wrong)}")
    a(f"- mean {syc_mean}  |  median {syc_median}")
    if have_hist:
        a(f"- histogram: `{os.path.basename(config.SYC_FRACTION_FIG_PATH)}`")
    a("")
    a("## Per-template stats")
    a("")
    a("| Template | Instances | Survival | Wrong-opinion rejected-class rate | Mean syc_fraction |")
    a("|---|---|---|---|---|")
    for tid, r in template_rows.items():
        a(f"| {tid} | {r['instances']} | {r['survival_rate']:.1%} | "
          f"{r['wrong_rejected_class_rate']:.2%} | {r['wrong_mean_syc_fraction']} |")
    if config.DROPPED_TEMPLATES:
        a("")
        a(f"Dropped templates (pilot gate 3): {', '.join(config.DROPPED_TEMPLATES)}")
    a("")
    a("## Grader")
    a("")
    if audit:
        a(f"- audit agreement: **{audit.get('agreement'):.1%}** on "
          f"{audit.get('n_labeled')} hand-labeled grades (required >= 95%)")
        a(f"- confusion (grader -> human): `{audit.get('confusion')}`")
    else:
        a("- audit: **PENDING** — run `audit_grader_sample.py --make` then hand-label "
          "and `--score` (blocking pilot gate 1)")
    a(f"- classifier API calls: {unparse['classifier_api_calls']} "
      f"(+ {unparse['classifier_cache_hits']} cache hits); "
      f"wrong-answer generator unique calls: {n_wrong_answer_calls}")
    a(f"- temperature: omitted (gpt-5-mini rejects non-default; spec deviation, "
      f"see manifest)")
    a("")
    a("## Parse health")
    a("")
    a(f"- UNPARSEABLE: {unparse['unparseable']} / {unparse['total_samples']} samples "
      f"({unparse['unparseable_rate']:.2%})")
    a(f"  - from truncation: {unparse['from_truncation']}")
    a(f"  - from missing final-answer line: {unparse['from_missing_final_answer_line']}")
    a(f"  - from invalid classifier output: {unparse['from_invalid_classifier_output']}")
    a("")
    a("## Final dataset")
    a("")
    a(f"- pairs (main): {len(pairs)}  |  wrong_opinion {n_wrong_final} / right_opinion "
      f"{len(pairs) - n_wrong_final}")
    a(f"- pairs (sham): {len(sham)}"
      + ("" if sham else "  (not built — pilot runs skip stage 10)"))
    a(f"- completion length (tokens): chosen mean {c_mean} / median {c_median}  |  "
      f"rejected mean {r_mean} / median {r_median}")
    if length_flag:
        a(f"- **FLAG**: chosen/rejected mean lengths differ by > 20% — check the "
          f"length-confound before training")
    a("")
    if gates:
        a("## Pilot gates")
        a("")
        for name, g in gates.items():
            a(f"- {name}: **{g['status']}**")
        a("")

    with open(config.QA_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"QA report -> {config.QA_REPORT_PATH}")


if __name__ == "__main__":
    main()
