from pathlib import Path
import re


_SAFE_DOMAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_NOTEBOOK_ARTIFACT_ROOT = Path("/kaggle/input/datasets/ali320230101/spectra-day1-2-artifacts-v1")


def accumulation_window(step: int, total_batches: int, grad_accum: int) -> tuple[int, int, bool]:
    if total_batches < 1 or grad_accum < 1 or not 0 <= step < total_batches:
        raise ValueError("Invalid gradient accumulation window")
    start = step // grad_accum * grad_accum
    return start, grad_accum, (step + 1) % grad_accum == 0


def generation_batch_size_for(requested: int | None, default: int) -> int:
    value = default if requested is None else requested
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("Generation batch size must be a positive integer")
    return value


def warmup_steps_for(total_steps: int) -> int:
    return max(1, total_steps // 10)


def format_example(prompt, response, tokenizer, max_length: int):
    import torch

    full_text = f"### Q:\n{prompt}\n\n### A:\n{response}{tokenizer.eos_token}"
    encoded = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    prefix = f"### Q:\n{prompt}\n\n### A:\n"
    prefix_length = len(tokenizer(prefix, truncation=True, max_length=max_length)["input_ids"])
    labels = input_ids.clone()
    labels[:prefix_length] = -100
    return {"input_ids": input_ids, "labels": labels, "attention_mask": encoded["attention_mask"][0]}


class PromptDataset:
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        return format_example(row["prompt"], row["response"], self.tokenizer, self.max_length)


def collate_fn(batch):
    import torch

    sequence_length = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.zeros(len(batch), sequence_length, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), sequence_length, dtype=torch.long)
    labels = torch.full((len(batch), sequence_length), -100, dtype=torch.long)
    for index, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[index, :length] = item["input_ids"]
        attention_mask[index, :length] = item["attention_mask"]
        labels[index, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def adapter_output_path(config, domain_name) -> Path:
    if not isinstance(domain_name, str) or not _SAFE_DOMAIN.fullmatch(domain_name):
        raise ValueError("domain_name must be a safe single path component")
    root = Path(config.work_dir)
    candidate = root / f"lora_{domain_name}"
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Adapter path escapes the configured work directory")
    return candidate


def _adapter_candidates(config, domain_name):
    name = f"lora_{domain_name}"
    return (
        Path(config.day12_dir) / name,
        _NOTEBOOK_ARTIFACT_ROOT / name,
        adapter_output_path(config, domain_name),
    )


def existing_adapter_path(config, domain_name):
    for candidate in _adapter_candidates(config, domain_name):
        if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def existing_training_adapter_path(config, domain_name):
    for candidate in _adapter_candidates(config, domain_name):
        if candidate.is_dir() and (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def train_domain_adapter(model, train_dataset, domain_name, config, logger, force=False):
    import gc

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

    saved_path = adapter_output_path(config, domain_name)
    existing = existing_training_adapter_path(config, domain_name)
    if existing is not None and not force:
        print(f"[LORA] Reusing {domain_name} at {existing}")
        return existing
    saved_path.mkdir(parents=True, exist_ok=True)
    print(f"[LORA] Training {domain_name} …")
    set_training_seed(config.seed)
    adapter_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model.model, adapter_config)
    peft_model.train()
    loader = DataLoader(
        PromptDataset(train_dataset, model.tokenizer, config.max_seq_len),
        batch_size=config.train_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in peft_model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    total_steps = max(1, (len(loader) // config.grad_accum) * config.train_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps_for(total_steps),
        total_steps,
    )
    for epoch in range(config.train_epochs):
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            _, _, should_step = accumulation_window(step, len(loader), config.grad_accum)
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = peft_model(**batch).loss / config.grad_accum
            loss.backward()
            total_loss += loss.item() * config.grad_accum
            steps += 1
            if should_step:
                trainable = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        print(f"  Epoch {epoch + 1}  loss={total_loss / max(1, steps):.4f}")
    peft_model.save_pretrained(saved_path, safe_serialization=True)
    model.tokenizer.save_pretrained(saved_path)
    del peft_model, optimizer, scheduler, loader
    if "batch" in locals():
        del batch
    if "loss" in locals():
        del loss
    if "trainable" in locals():
        del trainable
    old_model = model.model
    model.model = None
    del old_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    has_safetensors = any(Path(config.model_path).glob("*.safetensors"))
    model_options = {
        "dtype": getattr(torch, config.dtype),
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": True,
        "device_map": {"": model.device},
        "use_safetensors": has_safetensors,
    }
    if not has_safetensors:
        model_options["weights_only"] = True
    base_model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_options)
    model.replace_model(base_model)
    return saved_path


def set_training_seed(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
