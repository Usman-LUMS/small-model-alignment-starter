
"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    import torch

    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses must have the same length.")
    if max_length < 1:
        raise ValueError("max_length must be positive.")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("A pad or EOS token ID is required.")

    sequences = []
    label_sequences = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False
        )
        response_ids = tokenizer.encode(
            response, add_special_tokens=False
        )

        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]

        input_ids = (prompt_ids + response_ids)[:max_length]

        labels = (
            [-100] * len(prompt_ids) + response_ids
        )[:max_length]

        sequences.append(input_ids)
        label_sequences.append(labels)

    width = max((len(seq) for seq in sequences), default=0)

    padded_inputs = []
    padded_masks = []
    padded_labels = []

    for input_ids, labels in zip(sequences, label_sequences):
        padding_length = width - len(input_ids)

        padded_inputs.append(
            input_ids + [pad_id] * padding_length
        )
        padded_masks.append(
            [1] * len(input_ids) + [0] * padding_length
        )
        padded_labels.append(
            labels + [-100] * padding_length
        )

    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    import torch
    import torch.nn.functional as F

    batch = build_sft_batch(
        tokenizer, prompts, responses, max_length
    )

    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    token_logps = F.log_softmax(
        shift_logits.float(), dim=-1
    )

    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    selected_logps = torch.gather(
        token_logps,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    selected_logps = selected_logps.masked_fill(
        ~valid_mask, 0.0
    )

    return selected_logps.sum(dim=-1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    response_logps = compute_response_logprobs(
        model, tokenizer, prompts, responses, max_length
    )

    return -response_logps.mean()


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    chosen_batch = build_sft_batch(
        tokenizer,
        prompts,
        chosen_responses,
        max_length,
    )

    rejected_batch = build_sft_batch(
        tokenizer,
        prompts,
        rejected_responses,
        max_length,
    )

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    import torch.nn.functional as F

    policy_gap = (
        policy_chosen_logps - policy_rejected_logps
    )

    reference_gap = (
        reference_chosen_logps - reference_rejected_logps
    )

    logits = beta * (policy_gap - reference_gap)

    return -F.logsigmoid(logits).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    import torch

    policy_chosen_logps = compute_response_logprobs(
        policy_model,
        tokenizer,
        prompts,
        chosen_responses,
        max_length,
    )

    policy_rejected_logps = compute_response_logprobs(
        policy_model,
        tokenizer,
        prompts,
        rejected_responses,
        max_length,
    )

    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs(
            reference_model,
            tokenizer,
            prompts,
            chosen_responses,
            max_length,
        )

        reference_rejected_logps = compute_response_logprobs(
            reference_model,
            tokenizer,
            prompts,
            rejected_responses,
            max_length,
        )

    reference_chosen_logps = reference_chosen_logps.to(
        policy_chosen_logps.device
    )

    reference_rejected_logps = reference_rejected_logps.to(
        policy_rejected_logps.device
    )

    return compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
