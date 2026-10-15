# PA2 Submission Checklist

Audit date: 2026-10-15. Verdict: **BLOCKED** (required evaluation results are missing).

## Manual §16 deliverables

| # | Deliverable | Status | Location / evidence |
|---|---|---|---|
| 1 | GitHub repository URL | Ready once pushed | https://github.com/Usman-LUMS/small-model-alignment-starter (`main`) |
| 2 | Final report as PDF, ≤ 8 pages | **MISSING** | `report.md` exists but is incomplete; no PDF has been exported |
| 3 | Training plots for SFT and DPO | Partial | `figures/sft_loss.png`, `figures/hh_rlhf_dpo_training.png`; UltraFeedback has no history file, so no curve |
| 4 | Evaluation summary, DPO vs SFT | **MISSING** | No generations, pairwise JSONL or judgments exist |
| 5 | `submission_adapter.py` + public tests passing | OK | 6 signatures unchanged; `pytest tests/test_public_adapter_smoke.py`: 7 passed |

## Tasks

| Task | Status |
|---|---|
| 1 Repo + Kaggle setup | Done (2× T4 verified in notebooks) |
| 2 Baseline generation (AlpacaEval 300, seed 42) | Done: `results/baseline/` |
| 3 SFT, 30k | Done: val loss 1.4290; `best` == `final` |
| 4a HH-RLHF DPO, from scratch, 15k | Done, with deviations: final checkpoint only; resume re-sampled pairs |
| 4b UltraFeedback DPO, from scratch, 15k | **Deviation**: TRL + LoRA on 468 pairs |
| 5 Persona LoRA-DPO (SmolLM2) | **NOT RUN** |
| 6 SFT/DPO generations + pairwise files | **NOT RUN** |
| 7 Judge evaluation | **NOT RUN** |
| 8 Report | Incomplete draft (`report.md`) |

## Repository contents to commit

- `report.md`, `SUBMISSION_CHECKLIST.md`, `.gitignore` (adds `.venv/`, `*.egg-info/`, weights, `*.zip`, `outputs/`)
- `notebooks/Version1.ipynb`: baseline generation + SFT smoke run
- `notebooks/Version2.ipynb`: full SFT run (a different run from the shipped checkpoint) + DPO smoke test
- `notebooks/Version3.ipynb`: full SFT run matching the shipped checkpoint (val 1.4290417)
- `notebooks/Version5.ipynb`: HH/UltraFeedback DPO code, **not executed** (only 4 setup cells have outputs)
- `results/`, `figures/`: copied from the downloaded Kaggle artifacts

The executed HH-DPO run (`Downloads/notebook105a4f89ef (3).ipynb`) and the UltraFeedback TRL run (`notebook105a4f89ef (6).ipynb`) are not among the requested notebooks.

## Model weights (excluded from Git; not a §16 deliverable)

| Archive (Downloads) | SHA-256 | Manifest |
|---|---|---|
| `01_full_sft_model.zip` | `688eeb849893131a7fcd1ae6d68be1eba87c86f9852abc277d69fbe8f6c97d3b` | match |
| `02_hh_rlhf_dpo_model.zip` | `d478898105bb6d6ff497db0504a92e18b65dfc3cee18177853d2a183a2dbac0a` | match |
| `03_ultrafeedback_dpo_model.zip` | `7d31acf028b7726da79e54371ea73901db1adbb092218b5d06f7495150490cff` | not in manifest |
| `04_datasets_and_results.zip` | `e6129f78e8024b698999ed6bfa8d510d7358c3486b8e72628149ac918bdc9a16` | match |

Numbered copies `(1)`–`(5)` are byte-identical duplicates (same CRC listings). `pa2_sft_smoke.zip` comes from the smoke run and is superseded.
