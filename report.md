# PA2 — Small-Model Alignment: Supervised Fine-Tuning and Direct Preference Optimization

**Course:** Advanced Generative AI  
**Assignment:** PA2 — Small-Model Alignment  
**Base Model:** `Qwen/Qwen2.5-0.5B`  
**GitHub Repository:** https://github.com/Usman-LUMS/small-model-alignment-starter

## Abstract

This report presents supervised fine-tuning (SFT) and preference alignment of a small causal language model using Direct Preference Optimization (DPO). The experiments use Qwen2.5-0.5B as the initial checkpoint, UltraChat for supervised instruction tuning, Anthropic HH-RLHF for preference optimization, and UltraFeedback for an additional reduced-scale LoRA-DPO experiment.

The documented UltraFeedback experiment retained 468 training and 42 validation preference pairs after preprocessing and completed 59 optimizer steps. The resulting model achieved a validation DPO loss of 0.7441, reward accuracy of 66.67%, and a mean relative reward margin of +0.2336.

These results indicate a positive preference-ordering signal on a small validation set but do not independently demonstrate that DPO-generated responses outperform the SFT baseline. The report distinguishes experimentally observed findings from results that could not be independently verified from the available training and evaluation records.

## 1. Introduction

Pretrained language models acquire general linguistic and reasoning capabilities through large-scale next-token prediction. However, pretraining alone does not necessarily produce responses aligned with user instructions or preferences.

Supervised Fine-Tuning (SFT) adapts a pretrained model using instruction-response demonstrations. Direct Preference Optimization (DPO) subsequently optimizes the model using pairs of preferred and rejected responses, without requiring a separately trained reward model.

This assignment investigates the application of these techniques to a relatively small language model, Qwen2.5-0.5B.

The experimental work addresses:

1. Supervised instruction tuning using UltraChat.
2. Preference optimization using Anthropic HH-RLHF.
3. Additional preference optimization using UltraFeedback.
4. Evaluation of preference learning and model behavior.
5. Reproducibility and deployment considerations for trained checkpoints.

A separate persona-style LoRA-DPO task is also part of the assignment specification; its completed experimental results could not be verified from the available materials.

## 2. Repository Organization and Implementation

The implementation is maintained in the public GitHub repository:

**https://github.com/Usman-LUMS/small-model-alignment-starter**

The repository contains the assignment starter framework, model implementation code, Kaggle training notebooks, evaluation resources, and supporting documentation.

### 2.1 Repository directory structure

The project is organized around the following directory layout. The listing includes the notebook versions identified for submission; optional paths and exact filenames should be checked against the final pushed repository.

```text
small-model-alignment-starter/
│
├── README.md
├── report.md
├── submission_adapter.py
├── requirements.txt
│
├── notebooks/
│   ├── Version1.ipynb
│   ├── Version2.ipynb
│   ├── Version3.ipynb
│   └── Version5.ipynb
│
├── data/
├── docs/
├── eval_sets/
├── tests/
└── results/
```

### 2.2 Description of repository components

| Component | Purpose |
|---|---|
| `README.md` | Documents project setup, dependencies, and execution instructions. |
| `report.md` | Contains the experimental methodology, results, analysis, and limitations. |
| `submission_adapter.py` | Provides the interface connecting the implementation to the assignment evaluation workflow. |
| `notebooks/` | Contains Kaggle notebooks recording training experiments, development, and evaluation procedures. |
| `data/` | Contains dataset-related resources and preprocessing materials. |
| `docs/` | Contains supporting documentation. |
| `eval_sets/` | Contains resources for fixed evaluation prompts and comparisons. |
| `tests/` | Contains automated checks for implementation correctness. |
| `results/` | Intended for lightweight evaluation metrics, figures, and experimental outputs where available. |

### 2.3 Kaggle notebook organization

The Kaggle notebooks are maintained separately to preserve the experimental record.

Each notebook version represents a saved stage of the development and experimentation process. The existence of a notebook does not necessarily indicate that every experiment in that notebook completed successfully.

The notebook files are included to document the training workflow, dependency changes, configurations, execution records, and available output metrics.

### 2.4 Model checkpoint storage

Large trained model artifacts are stored separately from the ordinary Git repository to avoid including multi-gigabyte binary files in source control.

The locally retained archives include:

| Artifact | Description |
|---|---|
| `01_full_sft_model.zip` | Full supervised fine-tuned model checkpoint. |
| `02_hh_rlhf_dpo_model.zip` | HH-RLHF preference-optimized model checkpoint. |
| `03_ultrafeedback_dpo_model.zip` | PEFT/LoRA adapter produced by the reduced UltraFeedback experiment. |

The UltraFeedback adapter is not a standalone model and requires the compatible SFT base checkpoint and tokenizer.

Because these model archives are stored locally, their existence does not imply that they are directly downloadable from the public GitHub repository.

## 3. Models, Datasets, and Experimental Setup

### 3.1 Base language model

The experiments use `Qwen/Qwen2.5-0.5B`, a decoder-only causal language model.

The relatively small model size makes fine-tuning and preference optimization more feasible under constrained GPU resources.

### 3.2 Datasets

Three datasets were used across the documented workflows.

**UltraChat 200k**

The `HuggingFaceH4/ultrachat_200k` dataset contains conversational instruction-response examples and was used for supervised fine-tuning.

**Anthropic HH-RLHF**

The `Anthropic/hh-rlhf` dataset provides chosen and rejected responses reflecting human preferences concerning helpfulness and harmlessness.

**UltraFeedback Binarized**

The `trl-lib/ultrafeedback_binarized` dataset provides preferred and rejected conversational responses for preference optimization.

In the documented reduced UltraFeedback experiment, 550 preference pairs were streamed and shuffled using seed 42.

These were divided into:

- 500 intended training pairs.
- 50 intended validation pairs.

After preprocessing removed examples that were fully truncated, the effective dataset contained:

- **468 training pairs.**
- **42 validation pairs.**

This experiment was substantially smaller than the assignment's prescribed 15,000-pair from-scratch UltraFeedback track.

## 4. Training Methodology

### 4.1 Supervised Fine-Tuning

Supervised Fine-Tuning trains the language model to maximize the likelihood of target responses conditioned on instruction prompts.

For a dataset of instruction-response pairs, the objective minimizes the negative log-likelihood of the desired response tokens.

The project uses an UltraChat-based SFT workflow initialized from Qwen2.5-0.5B.

A full SFT model archive was saved locally. However, the available records do not establish the complete final optimizer configuration or checkpoint-selection procedure.

### 4.2 Direct Preference Optimization

DPO uses preferred and rejected responses to optimize a language model relative to a reference policy.

For each prompt, the preference dataset specifies:

- A chosen response.
- A rejected response.

The DPO objective encourages the policy to assign greater relative likelihood to preferred responses while controlling divergence from a reference model.

The HH-RLHF implementation uses an SFT policy and a frozen SFT reference model.

The completed training counts and external comparison results for this track were not independently established from the available records.

### 4.3 Reduced UltraFeedback LoRA-DPO experiment

The additional UltraFeedback experiment uses Hugging Face TRL's `DPOTrainer` and Parameter-Efficient Fine-Tuning through LoRA.

| Parameter | Value |
|---|---|
| Initial policy | SFT checkpoint |
| Effective training pairs | 468 |
| Effective validation pairs | 42 |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Maximum sequence length | 512 |
| Learning rate | 5 × 10⁻⁵ |
| DPO beta | 0.1 |
| Precision | FP16 |
| Data split seed | 42 |
| Execution device | `cuda:0` |

The LoRA configuration was:

| Parameter | Value |
|---|---|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Bias | None |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

The model contained 498,431,872 total parameters, of which 4,399,104 were trainable.

Therefore, approximately **0.8826% of model parameters** were updated during the LoRA training run.

Although the notebook environment exposed two NVIDIA T4 GPUs, the recorded execution on `cuda:0` does not establish multi-GPU distributed training.

### 4.4 Environment and dependency issues

The experiment encountered a compatibility issue between `torchao` 0.10.0 and PEFT 0.20.0.

Removing the conflicting `torchao` installation resolved the issue for the recorded run.

A structural check also confirmed that the trained model did not contain an unintended nested PEFT wrapper.

## 5. Experimental Results

### 5.1 UltraFeedback training behavior

The reduced UltraFeedback LoRA-DPO experiment completed **59 optimizer steps**.

Selected training-loss values were:

| Optimizer Step | Training Loss |
|---:|---:|
| 10 | 0.148299 |
| 20 | 0.180200 |
| 30 | 0.259071 |
| 40 | 0.283702 |
| 50 | 0.315139 |

The observed training loss generally increased over the selected interval.

However, this trend alone does not demonstrate that generated response quality became better or worse.

DPO loss measures a preference optimization objective, while the usefulness of generated responses requires separate evaluation on held-out prompts.

A complete validation-loss curve was not established from the recorded results.

### 5.2 UltraFeedback validation results

The model was evaluated using the 42 retained validation preference pairs.

| Metric | Result |
|---|---:|
| Validation DPO Loss | 0.744134 |
| Reward Accuracy | **0.666667 (66.67%)** |
| Mean Reward Margin | **+0.233630** |
| Mean Chosen Reward | −0.502133 |
| Mean Rejected Reward | −0.735763 |
| Mean Token Accuracy | 0.582261 |

The positive mean reward margin indicates that chosen responses received higher DPO-derived relative rewards on average than rejected responses.

The reward accuracy of 66.67% indicates that the preferred response received the higher relative reward for approximately two-thirds of the validation pairs.

These measurements provide evidence of preference discrimination within the small validation dataset.

However, they do not measure response generation quality relative to the SFT baseline.

### 5.3 DPO versus SFT evaluation

The assignment also requires generation-based comparisons between SFT and preference-optimized models.

A fixed AlpacaEval subset containing 300 prompts was prepared using seed 42.

The available evidence does not establish that all required paired model generations and blinded judge evaluations were completed.

| Evaluation | Status |
|---|---|
| HH-RLHF DPO versus SFT | No verified final paired-judge results |
| UltraFeedback DPO versus SFT | No verified final paired-judge results |
| Persona LoRA-DPO evaluation | No verified completed results |

Therefore, no numerical win, tie, or loss percentages are reported for these comparisons.

Importantly, the 66.67% internal reward accuracy must not be interpreted as a 66.67% DPO-versus-SFT win rate.

## 6. Discussion

The results demonstrate the feasibility of conducting a reduced-scale preference optimization experiment on a small language model using parameter-efficient adaptation.

By updating less than 1% of the model's parameters, LoRA substantially reduces the number of trainable weights relative to full-model fine-tuning.

The positive internal reward margin suggests that the optimized policy learned some degree of preference ordering on the retained validation examples.

Nevertheless, several factors limit the conclusions.

First, the experiment used only 468 effective training pairs, substantially fewer than the prescribed full-scale dataset.

Second, evaluation was performed on only 42 retained validation pairs.

Third, the experiment used one data split and one documented training run.

Finally, without completed generation-based comparisons, it is not possible to determine whether the DPO model produces consistently more helpful, harmless, or higher-quality responses than the SFT model.

The HH-RLHF and UltraFeedback datasets also encode different dimensions of response preference. HH-RLHF emphasizes helpful and harmless behavior, whereas UltraFeedback represents broader response-quality preferences.

The persona-style task measures stylistic adherence and should be evaluated independently.

## 7. Limitations and Reproducibility

The main limitations of the submitted work are summarized below.

**Dataset scale:** The recorded UltraFeedback experiment used 468 effective training pairs rather than the required 15,000-pair from-scratch DPO configuration.

**Evaluation completeness:** Complete blinded pairwise-judge results for SFT versus DPO were not verified.

**SFT and HH-RLHF records:** Although checkpoint archives were downloaded, their complete final configurations, checkpoint-selection evidence, and quantitative evaluation summaries were not independently confirmed.

**Persona experiment:** Completion of the independent SmolLM2 persona-style LoRA-DPO task could not be established.

**Statistical coverage:** The reported preference metrics come from a small validation split without repeated-seed or prompt-category analysis.

**Checkpoint accessibility:** Large model weights remain outside ordinary Git, and the public repository alone does not guarantee access to all trained checkpoints.

Reproducibility materials are organized through the GitHub repository, including implementation code, saved Kaggle notebooks, dataset resources, and the `submission_adapter.py` interface.

## 8. Conclusion

This assignment explores supervised fine-tuning and direct preference optimization for a small causal language model based on Qwen2.5-0.5B.

The documented supplementary UltraFeedback LoRA-DPO experiment successfully completed 59 optimizer steps using 468 effective training pairs and 42 validation pairs.

The resulting model achieved:

- **66.67% internal validation reward accuracy.**
- **+0.2336 mean relative reward margin.**
- **0.7441 validation DPO loss.**

These findings indicate preliminary preference-ordering capability on a small held-out sample.

They do not establish that the DPO model outperforms the supervised fine-tuned baseline in generated response quality.

The project illustrates both the practical feasibility of parameter-efficient preference optimization and the importance of dataset scale, controlled evaluation, reproducibility, and transparent reporting when assessing small-model alignment.

## References

1. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model.* arXiv:2305.18290.
2. Qwen Team. *Qwen2.5 Model Documentation.* Hugging Face.
3. Hugging Face. *TRL DPOTrainer Documentation.*
4. Hugging Face. *Parameter-Efficient Fine-Tuning (PEFT) Documentation.*
5. HuggingFaceH4. *UltraChat 200k Dataset.*
6. Anthropic. *HH-RLHF Dataset.*
7. TRL Library. *UltraFeedback Binarized Dataset.*
8. LUMS Advanced Generative AI. *PA2 — Small-Model Alignment: SFT and DPO*, course assignment handout.
