import csv
import gc
import io
import logging
import os
import random
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, content: str) -> None:
    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_csv(path: str | Path, rows, fieldnames) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def atomic_save_npy(path: str | Path, array) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    atomic_write_bytes(path, buffer.getvalue())


def configure_run(config, stage: str):
    output = Path(config.work_dir)
    directories = [output, output / "figures", output / "artifacts"]
    directories.extend(output / f"lora_{name}" for name in config.lora_names)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    datasets_cache = output / "hf_cache"
    hf_home = output / "hf_home"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HOME"] = str(hf_home)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        from datasets import disable_caching
    except ImportError:
        pass
    else:
        disable_caching()
    logger = logging.getLogger("spectra")
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vram_mb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def reset_vram():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gc_collect():
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Timer:
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.started = time.time()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.elapsed = time.time() - self.started
        print(f"[TIME] {self.label}: {self.elapsed:.2f}s")


def print_sep(title="", width=70, char="="):
    padding = max(0, width - len(title) - 2)
    print(f"\n{char * (padding // 2)} {title} {char * (padding - padding // 2)}" if title else char * width)
