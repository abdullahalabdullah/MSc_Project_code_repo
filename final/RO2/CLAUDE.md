# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Keep documentation in sync

Whenever a change alters how this repo is run, structured, or configured — new/removed/renamed
files, changed CLI flags or env vars, a different pipeline order, a new dependency — update the
relevant docs in the **same turn**, not as a follow-up:

- `README.md` (repo root), `harmlessness-DPO-train-dataset-curation/README.md`, and
  `anti-syc-DPO-train-dataset-curation/README.md` (modules)
- `DOCUMENTATION.md` (end-to-end runbook)
- `requirements.txt` if dependencies changed
- Any inline usage comments in `train.sh` / `train_antisyc.sh` /
  the modules' `run_all.sh` + `run_phase*.sh` that echo commands

Do not leave a documented command, path, or file listing that no longer matches the code. If a
file is deleted, grep the whole repo (code + docs) for its name and remove every reference, not
just the import.
