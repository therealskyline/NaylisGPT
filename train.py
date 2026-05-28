import os
import sys
import contextlib
import warnings
import gc
import math
import json
import time
import threading
import traceback
import argparse
import numpy as np
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

from datetime import datetime
from functools import partial
from typing import Optional, List

warnings.filterwarnings("ignore", category=RuntimeWarning, module="transformer_engine")

os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import torch
torch.set_float32_matmul_precision("high")

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import tomllib
except ImportError:
    import tomli as tomllib

_TE_AVAILABLE = False
_fp8_recipe   = None

try:
    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as _te_recipe

    _fp8_recipe = _te_recipe.DelayedScaling(
        margin            = 0,
        fp8_format        = _te_recipe.Format.HYBRID,
        amax_history_len  = 16,
        amax_compute_algo = "max",
    )
    _TE_AVAILABLE = True
    print(f"  Transformer Engine : FP8 DelayedScaling HYBRID ({getattr(transformer_engine, '__version__', '?')})")

except ImportError:
    print("  transformer_engine non installé — FP8 désactivé, entraînement en BF16")

from naylisgdn import NaylisGDN
from naylisgdn.optimizers import configure_optimizers
from naylisgdn.scheduler import WSDScheduler

parser = argparse.ArgumentParser(description="NaylisGDN Pretrain", add_help=False)
parser.add_argument("--config",     type=str,  default="config/480M.toml")
parser.add_argument("--batch-size", type=int,  default=None)
parser.add_argument("--hf-token",   type=str,  default=None)
parser.add_argument("--hf-repo",    type=str,  default=None)
_args, _ = parser.parse_known_args()


def _load_config(path: str) -> dict:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
    cfg["soft_cap"] = cfg.get("soft_cap") or None
    return cfg


CONFIG = _load_config(_args.config)
if _args.batch_size is not None:
    print(f"  CLI override : batch_size {CONFIG['batch_size']} → {_args.batch_size}")
    CONFIG["batch_size"] = _args.batch_size

_HF_TOKEN         = _args.hf_token
_HF_REPO          = _args.hf_repo or CONFIG.get("hf_repo", "")
_HF_PUSH_INTERVAL = CONFIG.get("hf_push_interval", 1800)

_FSDP_ENABLED = False
_LOCAL_RANK    = 0
_WORLD_SIZE    = 1

if torch.cuda.is_available() and torch.cuda.device_count() > 1 and "RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    _LOCAL_RANK   = int(os.environ.get("LOCAL_RANK", 0))
    _WORLD_SIZE   = dist.get_world_size()
    _FSDP_ENABLED = True
    torch.cuda.set_device(_LOCAL_RANK)
    print(f"  FSDP : rank={_LOCAL_RANK}/{_WORLD_SIZE}  GPUs={_WORLD_SIZE}")

device = f"cuda:{_LOCAL_RANK}" if torch.cuda.is_available() else "cpu"

_fp8_active_global = CONFIG.get("use_fp8", False) and _TE_AVAILABLE and _fp8_recipe is not None

if _LOCAL_RANK == 0:
    print("=" * 80)
    print("NaylisGDN — FP8 Hopper | Muon+MARS | WSD | Sequence Packing | FSDP")
    print("=" * 80)
    print(f"\nCONFIG : {_args.config}")
    print(f"  embed={CONFIG['embed_dim']}  layers={CONFIG['num_layers']}  heads={CONFIG['num_heads']}  kv={CONFIG['n_kv_heads']}")
    print(f"  packing={'ON' if CONFIG['use_packing'] else 'OFF'}  seq_len={CONFIG['max_seq_len']}")
    print(f"  FP8={'ON ✅ (DelayedScaling HYBRID)' if _fp8_active_global else 'OFF (BF16)'}")
    print(f"  FSDP={'ON ✅ (' + str(_WORLD_SIZE) + ' GPUs)' if _FSDP_ENABLED else 'OFF (single GPU)'}")
if device.startswith("cuda"):
    gpu_idx = _LOCAL_RANK if _FSDP_ENABLED else 0
    print(f"  GPU={torch.cuda.get_device_name(gpu_idx)}  VRAM={torch.cuda.get_device_properties(gpu_idx).total_memory/1e9:.0f}GB")
    cap = torch.cuda.get_device_capability(gpu_idx)
    print(f"  Compute capability: SM{cap[0]}{cap[1]}")
    if cap[0] < 9:
        print("  ⚠️  FP8 requiert SM90+ (H100/H200). GPU détecté < Hopper — FP8 ignoré.")
        _fp8_active_global = False


# Repo dataset pour les splits pretrain_data_NNN.bin
_HF_DATA_REPO  = "silyan/Naylis1-1.3B"
_N_CHUNKS      = 5          # pretrain_data_000.bin … pretrain_data_004.bin
_DATA_PREFIX   = "pretrain_data"

def _chunk_path(idx: int) -> str:
    return f"{_DATA_PREFIX}_{idx:03d}.bin"


def _compute_chunk_indices(data_file: str):
    """Calcule TRAIN_IDX, VAL_IDX et le nombre de séquences pour un fichier donné."""
    probe     = np.memmap(data_file, dtype=np.uint32, mode="r")
    total_tok = len(probe)
    del probe
    seq_len_1 = CONFIG["max_seq_len"] + 1
    n_seqs    = total_tok // seq_len_1
    idx       = np.arange(n_seqs)
    val_seqs  = max(min(CONFIG["val_tokens"] // seq_len_1, int(n_seqs * 0.05)), 1)
    train_idx = idx[val_seqs:]
    val_idx   = idx[:val_seqs]
    print(f"  {data_file}  →  {total_tok/1e9:.2f}B tokens  "
          f"train={len(train_idx):,}  val={len(val_idx):,} séquences")
    return train_idx, val_idx


def _hf_download_data(data_file: str):
    """Télécharge uniquement le fichier pretrain_data_NNN.bin manquant depuis HF."""
    if os.path.exists(data_file):
        return      # déjà présent localement
    token = _HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        print(f"  ⚠  Fichier absent et HF_TOKEN manquant — impossible de télécharger {data_file}")
        return
    try:
        from huggingface_hub import hf_hub_download
        fname = os.path.basename(data_file)
        print(f"\n  ↓ Téléchargement {fname} depuis {_HF_DATA_REPO} …")
        hf_hub_download(
            repo_id   = _HF_DATA_REPO,
            filename  = fname,
            repo_type = "dataset",
            token     = token,
            local_dir = ".",
        )
        print(f"  ✓ {fname} téléchargé ({os.path.getsize(data_file)/1e9:.1f} GB)")
    except Exception as e:
        print(f"  ✗ Téléchargement échoué ({data_file}) : {e}")


def _hf_download():
    """Télécharge le checkpoint depuis le repo HF (modèle, pas données)."""
    if not _HF_TOKEN or not _HF_REPO:
        print("  --hf-token / hf_repo absent : skip download HF (mode local)")
        return
    try:
        import shutil
        from huggingface_hub import snapshot_download
        print(f"\nHugging Face — download checkpoint depuis {_HF_REPO}")
        snapshot_download(
            repo_id         = _HF_REPO,
            repo_type       = "dataset",
            local_dir       = ".",
            token           = _HF_TOKEN,
            ignore_patterns = ["*.md", "*.txt", ".gitattributes",
                               "pretrain_data_*.bin"],  # ne pas retélécharger les données
        )
        model_dir = os.path.dirname(CONFIG["checkpoint_file"])
        os.makedirs(model_dir, exist_ok=True)
        pt_name   = os.path.basename(CONFIG["checkpoint_file"])
        json_name = pt_name.replace(".pt", "_info.json")
        for fname in (pt_name, json_name):
            src, dst = os.path.join(".", fname), os.path.join(model_dir, fname)
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"  Checkpoint déplacé : {src} → {dst}")
    except Exception as e:
        print(f"  WARN HF download : {e}")


def hf_push_checkpoint(local_pt_path: str, step: int, epoch: int):
    if not _HF_TOKEN or not _HF_REPO:
        return
    try:
        from huggingface_hub import HfApi
        api      = HfApi(token=_HF_TOKEN)
        pt_name  = os.path.basename(local_pt_path)
        api.upload_file(path_or_fileobj=local_pt_path, path_in_repo=pt_name,
                        repo_id=_HF_REPO, repo_type="dataset",
                        commit_message=f"checkpoint step={step:,} epoch={epoch}")
        json_path = local_pt_path.replace(".pt", "_info.json")
        if os.path.exists(json_path):
            api.upload_file(path_or_fileobj=json_path, path_in_repo=os.path.basename(json_path),
                            repo_id=_HF_REPO, repo_type="dataset",
                            commit_message=f"info step={step:,} epoch={epoch}")
        print(f"  HF push OK → {_HF_REPO}  step={step:,}")
    except Exception as e:
        print(f"  WARN HF push : {e}")


_hf_download()

# ── Estimation de TOTAL_STEPS pour le scheduler ───────────────────────────────
# On sonde le chunk 0 s'il existe ; sinon on estime depuis SPLIT_TOKENS = 10B.
_SPLIT_TOKENS_EST   = 10_000_000_000    # tokens par fichier (doit coller à tokens.py)
_seq_len_1          = CONFIG["max_seq_len"] + 1
_chunk0             = _chunk_path(0)

if os.path.exists(_chunk0):
    _probe0     = np.memmap(_chunk0, dtype=np.uint32, mode="r")
    _n_seqs0    = len(_probe0) // _seq_len_1
    del _probe0
else:
    _n_seqs0 = _SPLIT_TOKENS_EST // _seq_len_1

_val_seqs0           = max(min(CONFIG["val_tokens"] // _seq_len_1, int(_n_seqs0 * 0.05)), 1)
_train_seqs0         = _n_seqs0 - _val_seqs0
_batches_per_chunk   = math.ceil(_train_seqs0 / CONFIG["batch_size"])
STEPS_PER_CHUNK      = max(math.ceil(_batches_per_chunk / CONFIG["gradient_accumulation"]), 1)
TOTAL_STEPS          = STEPS_PER_CHUNK * _N_CHUNKS

if _LOCAL_RANK == 0:
    src = "sondé" if os.path.exists(_chunk0) else "estimé"
    print(f"\nTOTAL_STEPS = {STEPS_PER_CHUNK:,} steps/chunk × {_N_CHUNKS} chunks "
          f"= {TOTAL_STEPS:,}  ({src})")

print(f"\nLoading tokenizer...")
_tok_local    = "./tokenizer"
_tok_cfg_id   = CONFIG.get("model", "Qwen/Qwen2.5-0.5B")   # depuis [tokenizer] du TOML
_tok_src      = _tok_local if os.path.isdir(_tok_local) else _tok_cfg_id
tokenizer     = AutoTokenizer.from_pretrained(_tok_src, trust_remote_code=True)
print(f"  source : {_tok_src}")

_special_tokens = CONFIG.get("special_tokens", [])   # [] → rien à ajouter
if _special_tokens:
    _missing = [t for t in _special_tokens
                if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if _missing:
        tokenizer.add_special_tokens({"additional_special_tokens": _missing})
        print(f"  + Tokens spéciaux ajoutés : {_missing}")
        os.makedirs(_tok_local, exist_ok=True)
        tokenizer.save_pretrained(_tok_local)
        print(f"  ✓ Tokenizer sauvegardé → {_tok_local}/")
    else:
        print(f"  ✓ Tokens spéciaux déjà présents : {_special_tokens}")
else:
    print(f"  ✓ Aucun token spécial à ajouter (special_tokens=[])")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG["vocab_size"] = len(tokenizer)
print(f"  vocab={len(tokenizer)}  eos={tokenizer.eos_token_id}")


class ChunkSubset(Dataset):

    def __init__(self, data_path, idx, seq_len, pad_token_id):
        self.data_path    = data_path
        self._data        = None
        self.idx          = idx
        self.seq_len      = seq_len
        self.pad_token_id = pad_token_id

    def _get_data(self):
        if self._data is None:
            self._data = np.memmap(self.data_path, dtype=np.uint32, mode="r")
        return self._data

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        start = int(self.idx[i]) * (self.seq_len + 1)
        chunk = torch.from_numpy(self._get_data()[start : start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1].clone(), chunk[1:].clone()


class PackedChunkDataset(Dataset):

    def __init__(self, data_path, idx, seq_len, eos_token_id):
        self.data_path    = data_path
        self._data        = None
        self.idx          = idx
        self.seq_len      = seq_len
        self.eos_token_id = eos_token_id

    def _get_data(self):
        if self._data is None:
            self._data = np.memmap(self.data_path, dtype=np.uint32, mode="r")
        return self._data

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        start = int(self.idx[i]) * (self.seq_len + 1)
        block = torch.from_numpy(self._get_data()[start : start + self.seq_len + 1].astype(np.int64))
        return block[:-1].clone(), block[1:].clone()


def packed_collate_fn(batch, eos_token_id, seq_len):
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = torch.stack(ys)
    all_cu = [0]
    max_sl = 1
    for i in range(x.size(0)):
        seq     = x[i]
        eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) == 0:
            all_cu.append(all_cu[-1] + seq_len)
            max_sl = max(max_sl, seq_len)
        else:
            prev = 0
            for pos in eos_pos.tolist():
                l = pos - prev + 1
                if l > 0:
                    all_cu.append(all_cu[-1] + l)
                    max_sl = max(max_sl, l)
                prev = pos + 1
            remaining = seq_len - prev
            if remaining > 0:
                all_cu.append(all_cu[-1] + remaining)
                max_sl = max(max_sl, remaining)
    return x, y, torch.tensor(all_cu, dtype=torch.int32), max_sl


def _gpu_tflops() -> float:
    if not torch.cuda.is_available():
        return 1.0
    cap = torch.cuda.get_device_capability()
    if cap[0] >= 9:
        return 1979.0
    if cap[0] >= 8:
        return 312.0
    return 1.0


def _fp8_ctx(use_fp8: bool):
    if use_fp8 and _TE_AVAILABLE and _fp8_recipe is not None:
        return te.fp8_autocast(enabled=True, fp8_recipe=_fp8_recipe)
    return contextlib.nullcontext()


@torch.no_grad()
def run_benchmark(model, vocab_size, seq_len, batch_size, steps=20):
    model.eval()
    flops_per_fwd = 6 * sum(p.numel() for p in model.parameters()) * seq_len
    gpu_tflops    = _gpu_tflops()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if next(model.parameters()).dtype == torch.float32:
        model = model.to(torch.bfloat16)

    for _ in range(3):
        with _fp8_ctx(_fp8_active_global):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        with _fp8_ctx(_fp8_active_global):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_tokens   = batch_size * seq_len * steps
    tokens_per_sec = total_tokens / elapsed
    mfu            = (flops_per_fwd * batch_size * steps / elapsed) / (gpu_tflops * 1e12) * 100

    model.train()
    return {"tokens_per_sec": tokens_per_sec, "mfu_pct": mfu}


def print_benchmark(label, m):
    print(f"\n{'─'*60}")
    print(f"  BENCHMARK : {label}")
    print(f"  tokens/sec : {m['tokens_per_sec']:,.0f}")
    print(f"  MFU        : {m['mfu_pct']:.1f}%")
    print(f"{'─'*60}")


class CheckpointManager:

    def __init__(self, path: str):
        self.path             = path
        self._last_hf_push    = 0.0
        self._last_local_save = 0.0
        self._save_thread     = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _write(self, cp, info_snapshot, json_path, step, epoch):
        new_path = json_path + ".new"
        with open(new_path, "w") as f:
            json.dump(info_snapshot, f, indent=2, default=str)
        tmp = self.path + ".tmp"
        torch.save(cp, tmp)
        os.replace(tmp, self.path)
        os.replace(new_path, json_path)
        elapsed = time.time() - self._last_hf_push
        if _HF_TOKEN and _HF_REPO and elapsed >= _HF_PUSH_INTERVAL:
            hf_push_checkpoint(self.path, step, epoch)
            self._last_hf_push = time.time()

    def save_if_due(self, every_hours: float, model, optimizers, scheduler, metadata: dict) -> bool:
        """Sauvegarde si au moins `every_hours` heures se sont écoulées depuis la dernière.

        Retourne True si une sauvegarde a été déclenchée, False sinon.
        Le premier appel sauvegarde toujours (temps écoulé = ∞ par rapport à t=0).
        """
        if time.time() - self._last_local_save >= every_hours * 3600:
            self.save(model, optimizers, scheduler, metadata)
            self._last_local_save = time.time()
            return True
        return False

    def save(self, model, optimizers, scheduler, metadata: dict):
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()
        m               = model._orig_mod if hasattr(model, "_orig_mod") else model
        muon_opt, adamw_opt = optimizers
        info_snapshot   = {**metadata, "last_save": datetime.now().isoformat(), "config": CONFIG}
        cp = {
            "model_state_dict":     m.state_dict(),
            "muon_state_dict":      muon_opt.state_dict(),
            "adamw_state_dict":     adamw_opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metadata":             info_snapshot,
        }
        json_path = self.path.replace(".pt", "_info.json")
        print(f"  SAVE → epoch={metadata['current_epoch']}  step={metadata['global_step']:,}  [{self.path}] (async)")
        self._save_thread = threading.Thread(
            target=self._write,
            args=(cp, info_snapshot, json_path, metadata["global_step"], metadata["current_epoch"]),
            daemon=True,
        )
        self._save_thread.start()

    def wait(self):
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()

    def load(self):
        if not os.path.exists(self.path):
            return None
        print(f"\nCheckpoint trouvé : {self.path}")
        cp        = torch.load(self.path, map_location="cpu", weights_only=False)
        json_path = self.path.replace(".pt", "_info.json")
        new_path  = json_path + ".new"
        if os.path.exists(new_path):
            if os.path.exists(json_path):
                os.remove(json_path)
            os.replace(new_path, json_path)
        if os.path.exists(json_path):
            with open(json_path) as f:
                info = json.load(f)
            for k in ("global_step", "current_epoch", "epoch_start_step",
                      "skip_batches", "total_training_time", "training_history",
                      "actual_chunk_done", "current_chunk_idx"):
                default = 1 if k == "current_epoch" else (0.0 if k == "total_training_time" else 0)
                cp[k] = info.get(k, default)
        else:
            cp.update({"global_step": 0, "current_epoch": 1, "epoch_start_step": 0,
                       "skip_batches": 0, "total_training_time": 0.0,
                       "actual_chunk_done": 0, "current_chunk_idx": 0,
                       "training_history": {"validations": [], "epochs": []}})
        return cp


@torch.no_grad()
def validate(model, val_loader, max_batches=50):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    n          = 0
    ae, adt    = device.startswith("cuda"), torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast("cuda", dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y, pad_token_id=tokenizer.pad_token_id)
            total_loss += loss.detach()
            n += 1
    finally:
        model.train()
    avg = (total_loss / max(n, 1)).item()
    return math.exp(min(avg, 10)), avg


def train_epoch(model, optimizers, scheduler, checkpoint_manager, training_history,
                global_step, total_training_time, current_epoch, epoch_start_step,
                data_file, train_idx, val_idx,
                chunk_idx=0, actual_chunk_done=0, skip_batches=0):
    muon_opt, adamw_opt = optimizers
    label = f"Chunk {chunk_idx+1}/{_N_CHUNKS}  (pretrain_data_{chunk_idx:03d}.bin)"
    print(f"\n{'='*80}\n  {label}\n{'='*80}")

    if CONFIG["use_packing"]:
        train_ds = PackedChunkDataset(data_file, train_idx, CONFIG["max_seq_len"], tokenizer.eos_token_id)
    else:
        train_ds = ChunkSubset(data_file, train_idx, CONFIG["max_seq_len"], tokenizer.pad_token_id)
    val_ds = ChunkSubset(data_file, val_idx, CONFIG["max_seq_len"], tokenizer.pad_token_id)

    total_seqs = len(train_ds)
    if skip_batches >= math.ceil(total_seqs / CONFIG["batch_size"]):
        print("  Chunk déjà traité, skip.")
        return global_step, total_training_time, epoch_start_step

    _skip_samples = skip_batches * CONFIG["batch_size"]
    sampler       = torch.utils.data.SequentialSampler(range(_skip_samples, total_seqs))

    _collate = (
        partial(packed_collate_fn, eos_token_id=tokenizer.eos_token_id, seq_len=CONFIG["max_seq_len"])
        if CONFIG["use_packing"] else None
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], sampler=sampler,
        num_workers=CONFIG.get("num_workers", 4), pin_memory=True,
        persistent_workers=True, prefetch_factor=3, drop_last=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG.get("num_workers", 4), pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
    )

    total_batches = total_seqs // CONFIG["batch_size"]
    print(f"  train={total_batches:,} batches | restant={len(train_loader):,} | val={len(val_loader):,}")

    model.train()
    epoch_loss_t    = torch.zeros(1, device=device)
    running_loss_t  = torch.zeros(1, device=device)
    valid_batches   = 0
    running_batches = 0
    accumulated     = 0
    t_start         = time.time()
    ae, adt         = device.startswith("cuda"), torch.bfloat16 if device.startswith("cuda") else torch.float32

    pbar = tqdm(train_loader, desc=label, leave=True,
                initial=total_batches - len(train_loader), total=total_batches)

    for batch_idx, batch in enumerate(pbar):
        try:
            if CONFIG["use_packing"] and len(batch) == 4:
                x, y, cu_seqlens, max_seqlen = batch
                x          = x.to(device, non_blocking=True)
                y          = y.to(device, non_blocking=True)
                cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            else:
                x, y       = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                cu_seqlens = max_seqlen = None

            with _fp8_ctx(_fp8_active_global):
                with torch.amp.autocast("cuda", dtype=adt, enabled=ae):
                    _, loss, _ = model(
                        x, targets=y, pad_token_id=tokenizer.pad_token_id,
                        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                    )
                    loss = loss / CONFIG["gradient_accumulation"]

            loss.backward()
            accumulated += 1
            is_last = (batch_idx + 1 == len(train_loader))

            if accumulated % CONFIG["gradient_accumulation"] == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"], foreach=True)
                muon_opt.step()
                adamw_opt.step()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                scheduler.step()
                accumulated  = 0
                global_step += 1

                if global_step % CONFIG["validate_every_steps"] == 0:
                    val_ppl, val_loss = validate(model, val_loader, CONFIG["val_batches"])
                    avg = (running_loss_t / max(running_batches, 1)).item()
                    print(f"\n  step={global_step:,} | train={avg:.4f} ppl={math.exp(min(avg,10)):.1f} | "
                          f"val={val_loss:.4f} ppl={val_ppl:.1f} | lr={scheduler.get_last_lr()[0]:.2e}\n")
                    training_history["validations"].append({
                        "step": global_step, "current_epoch": current_epoch,
                        "val_loss": val_loss, "val_ppl": val_ppl,
                        "train_loss": avg, "lr": scheduler.get_last_lr()[0],
                    })
                    running_loss_t  = torch.zeros(1, device=device)
                    running_batches = 0

                checkpoint_manager.save_if_due(
                    CONFIG.get("save_every_hours", 1.0),
                    model, optimizers, scheduler,
                    metadata={
                        "current_epoch":       current_epoch,
                        "global_step":         global_step,
                        "epoch_start_step":    epoch_start_step,
                        "skip_batches":        batch_idx + 1,
                        "total_training_time": total_training_time + (time.time() - t_start),
                        "training_history":    training_history,
                        "actual_chunk_done":   actual_chunk_done,   # chunks entièrement terminés
                        "current_chunk_idx":   chunk_idx,           # chunk en cours
                    },
                )

            raw_t = loss.detach() * CONFIG["gradient_accumulation"]
            epoch_loss_t    += raw_t
            running_loss_t  += raw_t
            valid_batches   += 1
            running_batches += 1

            if batch_idx % 20 == 0:
                raw_f = raw_t.item()
                avg   = (running_loss_t / max(running_batches, 1)).item()
                pbar.set_postfix(
                    loss=f"{raw_f:.4f}", avg=f"{avg:.4f}",
                    ppl=f"{math.exp(min(avg,10)):.1f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}", step=f"{global_step:,}",
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM batch {batch_idx} — skip")
                torch.cuda.empty_cache()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                accumulated = 0
                gc.collect()
                model.train()
                continue
            raise

    pbar.close()
    elapsed             = time.time() - t_start
    total_training_time += elapsed
    avg_loss = (epoch_loss_t / max(valid_batches, 1)).item()
    print(f"\n  Epoch {current_epoch} terminée | loss={avg_loss:.4f} | {elapsed/60:.1f}min")
    training_history["epochs"].append({
        "epoch": current_epoch, "train_loss": avg_loss,
        "time_sec": elapsed, "global_step": global_step,
    })
    return global_step, total_training_time, epoch_start_step


def main():
    print("\n" + "=" * 80 + "\nCREATION MODELE\n" + "=" * 80)

    ckpt_mgr = CheckpointManager(CONFIG["checkpoint_file"])

    model = NaylisGDN(
        vocab_size=CONFIG["vocab_size"], embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"], num_layers=CONFIG["num_layers"],
        max_seq_len=CONFIG["max_seq_len"], dropout=CONFIG["dropout"],
        use_rope=CONFIG["use_rope"], rope_base=CONFIG.get("rope_base", 10000),
        use_yarn=CONFIG["use_yarn"],
        yarn_scale=CONFIG["yarn_scale"], yarn_original_max_len=CONFIG["yarn_original_max_len"],
        use_swiglu=CONFIG["use_swiglu"], n_kv_heads=CONFIG["n_kv_heads"],
        use_qk_norm=CONFIG["use_qk_norm"], soft_cap=CONFIG["soft_cap"],
        use_flash_attn=CONFIG["use_flash_attn"], use_fp8=_fp8_active_global,
        hybrid_ratio=CONFIG.get("hybrid_ratio", 3),
        gdn_head_dim=CONFIG.get("gdn_head_dim", None),
        gdn_v_heads=CONFIG.get("gdn_v_heads", None),
        attn_head_dim=CONFIG.get("attn_head_dim", None),
        use_moe=CONFIG.get("use_moe", False),
        num_experts=CONFIG.get("num_experts", 16),
        top_k_experts=CONFIG.get("top_k_experts", 2),
        shared_experts=CONFIG.get("shared_experts", 2),
        expert_hidden_dim=CONFIG.get("expert_hidden_dim", None),
        moe_aux_coeff=CONFIG.get("moe_aux_coeff", 0.01),
        use_moh=CONFIG.get("use_moh", False),
        moh_shared_heads=CONFIG.get("moh_shared_heads", None),
        moh_top_k_routed=CONFIG.get("moh_top_k_routed", None),
        mtp_steps=CONFIG.get("mtp_steps", 0),
        mtp_weight=CONFIG.get("mtp_weight", 0.3),
    ).to(device)

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.print_layout()

    total_params = sum(p.numel() for p in model.parameters())
    counts       = raw.count_parameters()
    print(f"\n  Params total : {total_params/1e6:.1f}M")
    print(f"    GDN blocks : {counts['gdn_blocks']/1e6:.1f}M  ({counts['num_gdn']} couches)")
    print(f"    GPT blocks : {counts['gpt_blocks']/1e6:.1f}M  ({counts['num_gpt']} couches)")
    print(f"  Précision    : {'FP8 (te.Linear + DelayedScaling HYBRID)' if _fp8_active_global else 'BF16 (nn.Linear)'}")

    if _FSDP_ENABLED:
        from naylisgdn.transformer_block import TransformerBlock
        from naylisgdn.gdn_block import GDNBlock
        _wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock, GDNBlock},
        )
        _mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=_wrap_policy,
            mixed_precision=_mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        print(f"  FSDP wrapping OK — ShardingStrategy.FULL_SHARD  {_WORLD_SIZE} GPU(s)")

    if device.startswith("cuda") and not _FSDP_ENABLED:
        bm0 = run_benchmark(model, CONFIG["vocab_size"], CONFIG["max_seq_len"],
                            min(CONFIG["batch_size"], 32), steps=CONFIG.get("benchmark_steps", 20))
        print_benchmark(f"Phase 0 — {'FP8' if _fp8_active_global else 'BF16'} no compile", bm0)

    if CONFIG.get("use_compile", True) and device.startswith("cuda"):
        import torch._dynamo
        import torch._inductor.config as inductor_cfg
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors  = True
        inductor_cfg.coordinate_descent_tuning             = True
        inductor_cfg.coordinate_descent_check_all_directions = True
        inductor_cfg.triton.unique_kernel_names            = True
        inductor_cfg.epilogue_fusion                       = True
        try:
            model = torch.compile(model, mode=CONFIG.get("compile_mode", "default"))
            print("  torch.compile : OK  (coordinate_descent_tuning ON)")
        except Exception as e:
            print(f"  torch.compile : FAIL ({e})")

    if device.startswith("cuda") and not _FSDP_ENABLED:
        dummy = torch.randint(0, CONFIG["vocab_size"], (4, CONFIG["max_seq_len"]), device=device)
        with _fp8_ctx(_fp8_active_global):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(dummy)
        torch.cuda.synchronize()
        bm1 = run_benchmark(model, CONFIG["vocab_size"], CONFIG["max_seq_len"],
                            min(CONFIG["batch_size"], 32), steps=CONFIG.get("benchmark_steps", 20))
        print_benchmark(f"Phase 1 — {'FP8' if _fp8_active_global else 'BF16'} + compile", bm1)
        speedup = bm1["tokens_per_sec"] / bm0["tokens_per_sec"]
        print(f"\n  Speedup compile : {speedup:.2f}x  |  MFU : {bm0['mfu_pct']:.1f}% → {bm1['mfu_pct']:.1f}%")

    raw_model  = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizers = configure_optimizers(
        raw_model, CONFIG["learning_rate"], CONFIG["weight_decay"],
        (CONFIG["adam_beta1"], CONFIG["adam_beta2"]), CONFIG["adam_eps"], device=device,
    )
    muon_opt, adamw_opt = optimizers

    scheduler = WSDScheduler(
        list(optimizers), max_lr=CONFIG["learning_rate"], total_steps=TOTAL_STEPS,
        warmup_ratio=CONFIG["warmup_ratio"], decay_ratio=CONFIG["decay_ratio"],
        min_lr_ratio=CONFIG["min_lr_ratio"],
    )

    training_history = {
        "config": CONFIG, "total_params": total_params,
        "total_steps": TOTAL_STEPS, "validations": [], "epochs": [],
        "start_time": datetime.now().isoformat(), "benchmarks": {},
    }

    # ── État initial ──────────────────────────────────────────────────────────
    global_step         = 0
    current_epoch       = 1
    epoch_start_step    = 0
    skip_batches        = 0
    total_training_time = 0.0
    actual_chunk_done   = 0     # nb de chunks entièrement terminés (0 = aucun)

    cp = ckpt_mgr.load()
    if cp:
        print("\nREPRISE")
        unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
        state = cp["model_state_dict"]
        state = {k: v for k, v in state.items() if not k.endswith("_extra_state")}
        unwrapped.load_state_dict(state, strict=False)
        if "muon_state_dict" in cp:
            muon_opt.load_state_dict(cp["muon_state_dict"])
            adamw_opt.load_state_dict(cp["adamw_state_dict"])
        scheduler.load_state_dict(cp["scheduler_state_dict"])
        global_step         = cp.get("global_step", 0)
        current_epoch       = cp.get("current_epoch", 1)
        epoch_start_step    = cp.get("epoch_start_step", 0)
        skip_batches        = cp.get("skip_batches", 0)
        total_training_time = cp.get("total_training_time", 0.0)
        training_history    = cp.get("training_history", training_history)
        actual_chunk_done   = cp.get("actual_chunk_done", 0)
        print(f"  chunk={actual_chunk_done}/{_N_CHUNKS}  step={global_step:,}  "
              f"skip_batches={skip_batches:,}")
        if actual_chunk_done >= _N_CHUNKS:
            print("Training déjà terminé sur tous les chunks.")
            return

    print(f"\n{'='*80}\nTRAINING START — {_N_CHUNKS} chunks de 10B tokens\n{'='*80}")

    # ── Boucle sur les chunks ─────────────────────────────────────────────────
    for chunk_idx in range(actual_chunk_done, _N_CHUNKS):
        chunk_file = _chunk_path(chunk_idx)
        print(f"\n{'─'*60}")
        print(f"  Chunk {chunk_idx+1}/{_N_CHUNKS} : {chunk_file}")
        print(f"{'─'*60}")

        # Télécharger si absent
        _hf_download_data(chunk_file)
        if not os.path.exists(chunk_file):
            print(f"  ✗ {chunk_file} introuvable et téléchargement impossible — STOP")
            break

        # Calculer les indices train/val pour ce chunk
        train_idx, val_idx = _compute_chunk_indices(chunk_file)

        # Reprise mid-chunk : skip_batches seulement pour le premier chunk
        _skip = skip_batches if chunk_idx == actual_chunk_done else 0
        if chunk_idx > actual_chunk_done:
            epoch_start_step = global_step

        try:
            global_step, total_training_time, epoch_start_step = train_epoch(
                model=model, optimizers=optimizers, scheduler=scheduler,
                checkpoint_manager=ckpt_mgr, training_history=training_history,
                global_step=global_step, total_training_time=total_training_time,
                current_epoch=current_epoch, epoch_start_step=epoch_start_step,
                data_file=chunk_file, train_idx=train_idx, val_idx=val_idx,
                chunk_idx=chunk_idx, actual_chunk_done=actual_chunk_done,
                skip_batches=_skip,
            )
        except KeyboardInterrupt:
            print("\nCTRL+C — sauvegarde en cours...")
            ckpt_mgr.save(model, optimizers, scheduler, metadata={
                "current_epoch": current_epoch, "global_step": global_step,
                "epoch_start_step": epoch_start_step, "skip_batches": 0,
                "total_training_time": total_training_time,
                "training_history": training_history,
                "actual_chunk_done": actual_chunk_done,
                "current_chunk_idx": chunk_idx,
            })
            ckpt_mgr.wait()
            return
        except Exception:
            print(f"\nERREUR chunk {chunk_idx}:\n{traceback.format_exc()}")
            ckpt_mgr.save(model, optimizers, scheduler, metadata={
                "current_epoch": current_epoch, "global_step": global_step,
                "epoch_start_step": epoch_start_step, "skip_batches": 0,
                "total_training_time": total_training_time,
                "training_history": training_history,
                "actual_chunk_done": actual_chunk_done,
                "current_chunk_idx": chunk_idx,
            })
            ckpt_mgr.wait()
            raise

        # ── Chunk terminé ────────────────────────────────────────────────────
        actual_chunk_done += 1
        skip_batches       = 0
        current_epoch     += 1
        print(f"\n  ✓ Chunk {chunk_idx+1}/{_N_CHUNKS} terminé  "
              f"(actual_chunk_done={actual_chunk_done})")

        ckpt_mgr.save(model, optimizers, scheduler, metadata={
            "current_epoch":     current_epoch,
            "global_step":       global_step,
            "epoch_start_step":  global_step,
            "skip_batches":      0,
            "total_training_time": total_training_time,
            "training_history":  training_history,
            "actual_chunk_done": actual_chunk_done,      # N chunks done
            "current_chunk_idx": actual_chunk_done,      # prochain chunk
        })
        ckpt_mgr.wait()

        hf_push_checkpoint(CONFIG["checkpoint_file"], global_step, current_epoch)

    print(f"\n{'='*80}\nTRAINING TERMINÉ — {actual_chunk_done}/{_N_CHUNKS} chunks\n{'='*80}")
    print(f"  Steps : {global_step:,}  Temps : {total_training_time/3600:.2f}h")

    history_path = CONFIG["checkpoint_file"].replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2, default=str)
    print(f"  History : {history_path}")
    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu")
    except Exception:
        print(traceback.format_exc())
    finally:
        if _FSDP_ENABLED and dist.is_initialized():
            dist.destroy_process_group()
        print("\nBye")
