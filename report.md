# PA2 — Small-Model Alignment: SFT and DPO

**Base checkpoint:** `Qwen/Qwen2.5-0.5B`
**Repository:** https://github.com/Usman-LUMS/small-model-alignment-starter
**Status:** Incomplete submission. Every value below comes from a saved artifact in `results/`, `figures/` or `notebooks/`. Items that were not run are marked **NOT RUN**; no result has been estimated or filled in.

## 1. Repository, models, and data

| Track | Data | Train / validation | Notes |
|---|---|---|---|
| SFT | `HuggingFaceH4/ultrachat_200k` (`train_sft`) | 30,000 / 1,000 | First user turn → first assistant turn; empty turns filtered; seed 42 |
| HH-RLHF DPO (from scratch) | `Anthropic/hh-rlhf` | 15,000 / 1,000 | Split at the final `\n\nAssistant:`; identical chosen/rejected pairs removed; seed 42 |
| UltraFeedback DPO | `trl-lib/ultrafeedback_binarized` (streamed, first 550 rows) | 500 / 50 requested → **468 / 42** after TRL dropped fully truncated rows | **TRL `DPOTrainer` + LoRA, not from scratch** (see §3) |
| Persona LoRA-DPO | `data/persona_lora/ramsay_style_preferences.json`, SmolLM2-360M-Instruct | — | **NOT RUN** |

**Prompt template (SFT, DPO and generation):** `### Instruction:\n{instruction}\n\n### Response:\n`. EOS (`<|endoftext|>`) is appended to each response and also used as the padding token.

**Fixed evaluation sets:** AlpacaEval, 300 prompts, seed 42 (`results/baseline/alpaca_eval_300_seed42.json`); synthetic helpful/harmless set (`eval_sets/hh_rlhf_eval_300_seed42.json`).

**`compute_sft_loss` reduction:** mean over examples of the per-example *summed* response negative log-probability (`submission_adapter.py`).

## 2. Experimental configuration (values actually used)

| | SFT | HH-RLHF DPO | UltraFeedback DPO |
|---|---|---|---|
| Parameters trained | Full fine-tuning | Full (policy) | LoRA r=8, α=16, dropout 0.05; q/k/v/o/gate/up/down proj; 4,399,104 trainable (0.88%) |
| Initial policy / reference | Qwen2.5-0.5B | SFT `best` / frozen SFT `best` (fp16, `cuda:1`) | SFT `best` / implicit reference (adapter disabled) |
| Optimizer, LR | AdamW, 2e-5 | AdamW, 1e-6 fixed | TRL default (AdamW), 5e-5 |
| Schedule / warmup | Cosine, 3% (28 steps) | None | TRL default |
| Weight decay / grad clip | 0.1 / 1.0 | 0.0 / 1.0 | TRL default |
| Batch × accumulation = effective batch | 2 × 16 = 32 | 1 × 32 = 32 | 1 × 8 = 8 |
| Precision | fp16 autocast, fp32 weights, GradScaler | fp16 autocast, fp32 policy | fp16 |
| Max length (prompt) | 512 | 512 (256) | 512 |
| Steps / epochs | 937 / 1 | 469 updates / 1 pass (see §3) | 59 / 1 |
| β | — | 0.1 | 0.1 |
| Seed | 42 | 42 | 42 (data split) |

**Generation settings:** baseline AlpacaEval generation used greedy decoding (`do_sample=False`), `max_new_tokens=128`, no repetition penalty. No SFT or DPO evaluation generations were produced.

## 3. Parameter choices and deviations

| Item | Guide | Used | Type | Effect / evidence |
|---|---|---|---|---|
| UltraFeedback method | From-scratch DPO | TRL `DPOTrainer` + LoRA | Resource (Kaggle GPU quota) | Not comparable with the from-scratch track |
| UltraFeedback size | 15,000 / 1,000 pairs | 468 / 42 | Resource | 42 validation pairs give very wide uncertainty |
| UltraFeedback LR | 1e-6 | 5e-5 | Resource (only 59 steps available) | Not ablated |
| HH-DPO checkpoint selection | Best by validation reward accuracy, plus final | Final only | Process gap | No `best` checkpoint exists |
| HH-DPO data order after resume | One pass over 15,000 pairs | Resumed at update 300 (9,600 pairs); the remaining 5,400 pairs were sampled from **all** 15,000 (`rng.sample`, seed 342) | Bug | Expected ≈3,456 repeats (64%). Train loss fell from ≈0.68 to ≈0.50 after the resume while validation did not improve (Fig. 2) |
| Persona LoRA-DPO | Required | NOT RUN | — | — |

## 4. Training behavior

- **SFT** (`figures/sft_loss.png`, `results/sft/metrics.jsonl`): validation loss decreased monotonically, 1.4831 (step 100) → 1.4290 (step 937); train loss 1.458 → 1.408. Because validation loss never rose, `best` and `final` are identical (same weight CRC). Runtime was 103.9 min.
- **HH-RLHF DPO** (`figures/hh_rlhf_dpo_training.png`, `results/hh_rlhf_dpo/`): train loss stayed near ln 2 for the first 9,600 pairs, then dropped abruptly at the resume point, and batch reward accuracy rose to ≈0.8. The final checkpoint on 1,000 validation pairs: DPO loss **0.7053** (above ln 2 = 0.6931), reward accuracy **0.555**, mean margin **+0.0789**. The train/validation gap is consistent with re-exposure to training pairs after the resume (§3), not with general preference learning.
- **UltraFeedback LoRA-DPO** (`results/ultrafeedback_lora_dpo/`): mean train loss 0.234, final batch reward accuracy 0.94. On 42 validation pairs: loss 0.744, reward accuracy 0.667, margin +0.234. No per-step history file was saved, so there is no curve.

## 5. DPO-versus-SFT results

| Evaluation set | DPO vs SFT | Judged pairs | W / T / L | Judge | Status |
|---|---|---|---|---|---|
| Synthetic helpful/harmless | HH-RLHF DPO vs SFT | — | — | — | **NOT RUN** |
| AlpacaEval 300, seed 42 | UltraFeedback DPO vs SFT | — | — | — | **NOT RUN** |
| Persona style prompts | LoRA-DPO vs SmolLM2 base | — | — | — | **NOT RUN** |

The only completed measurements are the internal DPO validation metrics in §4. These are preference-discrimination scores against the frozen reference; they are **not** judge-based win rates.

## 6. Interpretation and error analysis

Measured: the HH-RLHF policy shows only a weak held-out preference signal (accuracy 0.555, margin +0.079, loss above ln 2). Interpretation: the resume bug likely produced overfitting to repeated pairs, and the small fixed LR of 1e-6 limited movement during the first 300 updates. The UltraFeedback run shows a positive signal on 42 pairs, but the sample is too small to support conclusions. Without paired generations and judgments, no claim can be made about which model is better on either evaluation set, and no qualitative examples are reported.

## 7. Limitations and reproducibility

1. No DPO-vs-SFT generations or judgments exist (Tasks 6–7).
2. Persona LoRA-DPO (Task 5) was not run.
3. UltraFeedback deviates from the required from-scratch, 15k-pair setup.
4. HH-DPO has no `best` checkpoint, and the post-resume data sampling repeated training pairs.
5. Single seed, single run per track.
6. `notebooks/` holds Version1, 2, 3 and 5. Version3 contains the executed full SFT run. Version5 contains the HH-DPO code without executed outputs. The executed HH-DPO run and the UltraFeedback TRL run are in Kaggle notebook versions that are not included here.

**Artifact paths:** `results/sft/metrics.jsonl`; `results/hh_rlhf_dpo/{training_history,validation_results}.json`; `results/ultrafeedback_lora_dpo/*.json`; `results/baseline/*.json`; `figures/*.png`. Model weights are kept outside Git (see `SUBMISSION_CHECKLIST.md` for SHA-256 hashes).

## References

1. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
2. Bai et al., *Training a Helpful and Harmless Assistant with RLHF*, 2022.
3. Hugging Face Transformers, TRL and PEFT documentation; dataset cards for UltraChat-200k, HH-RLHF and UltraFeedback-binarized.
