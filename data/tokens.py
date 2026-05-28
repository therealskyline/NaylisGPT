"""
dataset/tokens.py — Tokenisation + curriculum assembly du mix 50B tokens
─────────────────────────────────────────────────────────────────────────
Pipeline :
  1. Pre-flight  : vérifie l'accès à chaque dataset (1 doc) avant tout download
  2. Download    : streaming → chunks binaires de ~1B tokens
                   chunks/{dataset_name}/chunk_000.bin, chunk_001.bin, ...
  3. Assembly    : 5 fichiers de 10B tokens chacun (~40 GB) :
                   pretrain_data_000.bin … pretrain_data_004.bin
                   Curriculum en 2 phases réparti sur les 2 fichiers :

       Phase 1 (0 → 30 % tokens)  : Cosmopedia-v2 seul, séquentiel
                                     → base syntaxique anglaise propre
       Phase 2 (30 → 100 % tokens): Tous datasets shufflés ensemble
                                     EN + Code mélangés
                                     → généralisation multi-domaine

  Chaque fichier est uploadé sur HuggingFace Hub dès sa clôture.

  Source                                          Phase   Volume   Langue
  ──────────────────────────────────────────────────────────────────────
  HuggingFaceTB/smollm-corpus [cosmopedia-v2]       1     15.0B    EN
  nv-community/Nemotron-CC-v2.1 [Non-Synth HQ]      2     10.0B    EN
  tokyotech-llm/swallow-code-v2 [Python, no-JP]     2      7.0B    Code
  HuggingFaceFW/finephrase [all]                     2      7.0B    EN
  nv-community/Nemotron-CC-Math-v1 [4plus]           2      6.0B    EN
  nv-community/Nemotron-CC-v2.1 [High-Synth]         2      5.0B    EN
  ──────────────────────────────────────────────────────────────────────
  TOTAL                                                    50.0B

Ratio linguistique approximatif :
  EN  : ~43B  (~86 %)
  Code:  ~7B  (~14 %)

Tokenizer : Qwen2.5-0.5B (vocab ~151 667, uint32)
            + special tokens <think> et </think>
            Sauvegardé localement dans ./tokenizer/ après le premier chargement
Reprise   : chunks/resume.json   (chunk courant + offset doc par dataset)
Usage     : python tokens.py [--reset] [--skip-assembly] [--only-assembly]
            python tokens.py --no-preflight   (saute le pre-flight)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("✗ tqdm non installé : pip install tqdm")

try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("✗ transformers non installé : pip install transformers")

# ── Tokens & identifiants ──────────────────────────────────────────────────────

TOKENIZER_ID    = "Qwen/Qwen2.5-0.5B"
TOKENIZER_LOCAL = Path("tokenizer")          # sauvegarde locale après ajout tokens

# vocab Qwen2.5 = 151 643 base + 22 spéciaux + 2 ajoutés = 151 667
# uint32 obligatoire : 151 667 > 65 535 (limite uint16)
DTYPE = np.uint32

# EOS chargé dynamiquement depuis le tokenizer (= 151 643 chez Qwen2.5)
# défini globalement après load_tokenizer()
EOS_ID: int = 151643   # valeur par défaut, écrasée au chargement

HF_TOKEN         = os.environ.get("HF_TOKEN", "")
MODELSCOPE_TOKEN = os.environ.get("MODELSCOPE_TOKEN", "")

# ── Chemins ────────────────────────────────────────────────────────────────────

CHUNKS_DIR  = Path("chunks")
RESUME_FILE = CHUNKS_DIR / "resume.json"

# Fichiers de sortie finaux : pretrain_data_000.bin … pretrain_data_007.bin
OUT_PREFIX  = "pretrain_data"    # préfixe des fichiers de sortie

# ── Paramètres de découpe ──────────────────────────────────────────────────────

CHUNK_TOKENS   = 1_000_000_000   # 1B tokens par chunk intermédiaire (= 4 GB uint32)
WRITE_BUF_TOKS = 25_000_000      # buffer d'écriture interne (100 MB en uint32)

# Taille d'un fichier final de pretrain (10B tokens = ~40 GB en uint32)
SPLIT_TOKENS   = 10_000_000_000

# Repo HuggingFace où uploader les fichiers finaux après assembly
HF_DATASET_REPO = "silyan/Naylis1-1.3B"

# ── Curriculum 3 phases ────────────────────────────────────────────────────────
#
#  Phase 1 : 0  → P1_END_FRAC   des tokens totaux  → Cosmopedia pur
#  Phase 2 : P1_END_FRAC → P3_START_FRAC           → Tout shufflé
#  Phase 3 : P3_START_FRAC → 1.0                   → Replay EN HQ
#
#  Datasets "replay" Phase 3 : nemotron_non_synth + finephrase
#  Proportion replay dans Phase 3 : REPLAY_FRAC des tokens Phase 3
#  (le reste = Phase 2 shufflé qui continue)

P1_END_FRAC   = 0.30    # Phase 1 = premiers 30 % des tokens (15B / 50B)
P3_START_FRAC = 1.00    # Pas de Phase 3 (replay désactivé)

# Pas de replay Phase 3 pour ce run 50B
REPLAY_DATASETS    = set()
REPLAY_RATIO_START = 0.0
REPLAY_RATIO_END   = 0.0

TEXT_FIELDS = ["text", "code", "content", "document", "passage", "input", "problem"]

# Détection japonais (hiragana + katakana + CJK) — utilisé pour skip_jp
import re as _re
_JP_RE = _re.compile(
    r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF'
    r'\uF900-\uFAFF\u20000-\u2A6DF]'
)

# ── Qualité FineWeb2-HQ ────────────────────────────────────────────────────────
# Filtrage par quality_score (0.0 → 1.0). None = pas de filtre.
FINEWEB2_MIN_QUALITY = 0.7   # ne garder que les docs ≥ 0.7

# ── Définition des datasets ────────────────────────────────────────────────────

DATASETS: list[dict] = [

    # ── Phase 1 : base syntaxique anglaise synthétique ────────────────────────
    {
        "name"        : "cosmopedia_v2",
        "hf_id"       : "HuggingFaceTB/smollm-corpus",
        "subset"      : "cosmopedia-v2",
        "split"       : "train",
        "token_target": 15_000_000_000,
        "phase"       : 1,
        "use_hf"      : True,
    },

    # ── Phase 2 : anglais web haute qualité (non-synthétique) ────────────────
    {
        "name"        : "nemotron_non_synth",
        "hf_id"       : "nv-community/Nemotron-CC-v2.1",
        "subset"      : "High-Quality",
        "split"       : "train",
        "token_target": 10_000_000_000,
        "phase"       : 2,
    },

    # ── Phase 2 : code Python anglais (kanji/kana exclus) ────────────────────
    {
        "name"        : "swallow_code",
        "hf_id"       : "tokyotech-llm/swallow-code-v2",
        "subset"      : "stage5-auto-format",
        "split"       : "train",
        "token_target": 7_000_000_000,
        "phase"       : 2,
        "lang_filter" : "python",
        "skip_jp"     : True,
        "use_hf"      : True,
    },

    # ── Phase 2 : anglais reformulé synthétique ───────────────────────────────
    {
        "name"        : "finephrase",
        "hf_id"       : "HuggingFaceFW/finephrase",
        "subset"      : "all",
        "split"       : "train",
        "token_target": 7_000_000_000,
        "phase"       : 2,
        "use_hf"      : True,
    },

    # ── Phase 2 : maths ───────────────────────────────────────────────────────
    {
        "name"        : "nemotron_math",
        "hf_id"       : "nv-community/Nemotron-CC-Math-v1",
        "subset"      : "4plus",
        "split"       : "train",
        "token_target": 6_000_000_000,
        "phase"       : 2,
    },

    # ── Phase 2 : anglais synthétique haute qualité ───────────────────────────
    {
        "name"        : "nemotron_high_synth",
        "hf_id"       : "nv-community/Nemotron-CC-v2.1",
        "subset"      : "High-Quality-Synthetic",
        "split"       : "train",
        "token_target": 5_000_000_000,
        "phase"       : 2,
    },
]

# ── Tokenizer ──────────────────────────────────────────────────────────────────

def load_tokenizer() -> AutoTokenizer:
    global EOS_ID

    # Charger depuis la sauvegarde locale si elle existe (plus rapide)
    src = TOKENIZER_LOCAL if TOKENIZER_LOCAL.exists() else TOKENIZER_ID
    print(f"Chargement tokenizer {'(local)' if TOKENIZER_LOCAL.exists() else TOKENIZER_ID}…")

    try:
        tok = AutoTokenizer.from_pretrained(
            str(src),
            trust_remote_code=True,
            token=HF_TOKEN or None,
        )
    except Exception as e:
        sys.exit(f"✗ Impossible de charger le tokenizer : {e}")

    # Ajouter <think> et </think> si pas déjà présents
    think_tokens = ["<think>", "</think>"]
    missing = [t for t in think_tokens
               if tok.convert_tokens_to_ids(t) == tok.unk_token_id]

    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
        print(f"  + Tokens ajoutés : {missing}")
        # Sauvegarder localement pour les prochaines reprises
        TOKENIZER_LOCAL.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(TOKENIZER_LOCAL))
        print(f"  ✓ Tokenizer sauvegardé → {TOKENIZER_LOCAL}/")
    else:
        print(f"  ✓ Tokens <think></think> déjà présents")

    # Mettre à jour EOS_ID global
    EOS_ID = tok.eos_token_id
    if EOS_ID is None:
        EOS_ID = tok.convert_tokens_to_ids("<|endoftext|>")

    think_open  = tok.convert_tokens_to_ids("<think>")
    think_close = tok.convert_tokens_to_ids("</think>")

    print(f"  vocab total  : {len(tok):,}")
    print(f"  EOS          : {tok.eos_token!r} = {EOS_ID}")
    print(f"  <think>      : {think_open}")
    print(f"  </think>     : {think_close}")
    print(f"  dtype        : uint32  (4 octets/token, ~800 GB pour 200B tokens)")

    return tok


# ── Accès aux datasets ─────────────────────────────────────────────────────────

def _get_text(row) -> str:
    """Extrait le texte d'une row, qu'elle soit dict ou str."""
    if isinstance(row, str):
        return row
    for f in TEXT_FIELDS:
        if f in row and row[f]:
            return str(row[f])
    vals = [str(v) for v in row.values() if isinstance(v, str) and len(v) > 10]
    return vals[0] if vals else ""


def _load_hf_stream(cfg: dict, doc_offset: int = 0):
    from datasets import load_dataset as hf_load
    kwargs: dict = dict(
        path      = cfg["hf_id"],
        split     = cfg["split"],
        streaming = True,
    )
    if cfg.get("subset"):
        kwargs["name"] = cfg["subset"]
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    ds = hf_load(**kwargs)
    if doc_offset:
        ds = ds.skip(doc_offset)
    return ds


def stream_dataset(cfg: dict, doc_offset: int = 0) -> Iterator[str]:
    lang         = cfg.get("lang_filter", "").strip().lower()
    min_quality  = cfg.get("quality_filter")   # float ou None
    skip_jp      = cfg.get("skip_jp", False)   # True → exclut les docs avec kanji/kana

    def _pass(row) -> bool:
        # filtre langue
        if lang and isinstance(row, dict):
            row_lang = (
                row.get("language") or row.get("lang")
                or row.get("programming_language") or ""
            )
            if str(row_lang).strip().lower() != lang:
                return False
        # filtre qualité FineWeb2-HQ
        if min_quality is not None and isinstance(row, dict):
            score = row.get("quality_score")
            if score is not None and float(score) < min_quality:
                return False
        return True

    def _no_jp(text: str) -> bool:
        return not _JP_RE.search(text)

    if cfg.get("use_hf"):
        try:
            ds = _load_hf_stream(cfg, doc_offset)
            for row in ds:
                if _pass(row):
                    t = _get_text(row)
                    if t and (not skip_jp or _no_jp(t)):
                        yield t
        except ImportError:
            sys.exit("✗ pip install datasets")
    else:
        # ModelScope
        try:
            from modelscope import MsDataset
        except ImportError:
            sys.exit("✗ pip install modelscope")
        kwargs: dict = dict(dataset_name=cfg["hf_id"], split=cfg["split"])
        if cfg.get("subset"):
            kwargs["subset_name"] = cfg["subset"]
        if MODELSCOPE_TOKEN:
            kwargs["token"] = MODELSCOPE_TOKEN
        try:
            ds = MsDataset.load(**kwargs, use_streaming=True)
        except TypeError:
            ds = MsDataset.load(**kwargs)
        for i, row in enumerate(ds):
            if i < doc_offset:
                continue
            if _pass(row):
                t = _get_text(row)
                if t and (not skip_jp or _no_jp(t)):
                    yield t


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def preflight_check(datasets: list[dict]) -> bool:
    print("\n" + "═" * 60)
    print("  PRE-FLIGHT — vérification accès datasets")
    print("═" * 60)
    all_ok = True

    for cfg in tqdm(datasets, desc="Pre-flight", unit="dataset"):
        name = cfg["name"]
        try:
            gen = stream_dataset(cfg, doc_offset=0)
            doc = next(gen, None)
            if doc is None:
                print(f"  ✗ {name} : stream vide (aucun document retourné)")
                all_ok = False
            else:
                preview = doc[:60].replace("\n", " ")
                print(f"  ✓ {name:<30}  → \"{preview}…\"")
        except Exception as e:
            print(f"  ✗ {name} : {e}")
            all_ok = False

    print("═" * 60)
    if all_ok:
        print("  ✓ Tous les datasets sont accessibles — download autorisé\n")
    else:
        print("  ✗ Certains datasets sont inaccessibles — ABANDON\n")
    return all_ok


# ── Reprise ────────────────────────────────────────────────────────────────────

def load_resume() -> dict:
    if RESUME_FILE.exists():
        with open(RESUME_FILE) as f:
            data = json.load(f)
        data.setdefault("failed", [])
        return data
    return {"completed": [], "in_progress": {}, "failed": []}


def save_resume(state: dict):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESUME_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Écriture chunked ───────────────────────────────────────────────────────────

class ChunkWriter:
    """
    Écrit un flux de tokens en fichiers chunk_NNN.bin dans `out_dir`.
    Chaque fichier contient exactement CHUNK_TOKENS tokens (uint32),
    sauf éventuellement le dernier.
    """

    def __init__(self, out_dir: Path, start_chunk: int = 0, chunk_offset: int = 0):
        self.out_dir    = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_idx  = start_chunk
        self.chunk_toks = chunk_offset
        self.buf: list  = []
        self.buf_size   = 0
        self._open_chunk()

    def _chunk_path(self) -> Path:
        return self.out_dir / f"chunk_{self.chunk_idx:03d}.bin"

    def _open_chunk(self):
        mode   = "ab" if self.chunk_toks > 0 else "wb"
        self.f = open(self._chunk_path(), mode)

    def _flush_buf(self):
        if self.buf:
            np.array(self.buf, dtype=DTYPE).tofile(self.f)
            self.f.flush()
            self.buf      = []
            self.buf_size = 0

    def write(self, tokens: list[int]):
        pos = 0
        while pos < len(tokens):
            space           = CHUNK_TOKENS - self.chunk_toks
            take            = min(space, len(tokens) - pos)
            self.buf.extend(tokens[pos : pos + take])
            self.buf_size  += take
            self.chunk_toks += take
            pos             += take

            if self.buf_size >= WRITE_BUF_TOKS:
                self._flush_buf()

            if self.chunk_toks >= CHUNK_TOKENS:
                self._flush_buf()
                self.f.close()
                self.chunk_idx  += 1
                self.chunk_toks  = 0
                self._open_chunk()

    def close(self) -> tuple[int, int]:
        self._flush_buf()
        self.f.close()
        return self.chunk_idx, self.chunk_toks

    @property
    def total_chunks_closed(self) -> int:
        return self.chunk_idx


# ── Download d'un dataset ──────────────────────────────────────────────────────

def download_dataset(cfg: dict, tokenizer, state: dict):
    name         = cfg["name"]
    token_target = cfg["token_target"]

    prog         = state.get("in_progress", {}).get(name, {})
    start_chunk  = prog.get("chunk_idx",   0)
    chunk_offset = prog.get("chunk_toks",  0)
    doc_offset   = prog.get("doc_offset",  0)
    tokens_done  = prog.get("tokens_done", 0)

    out_dir = CHUNKS_DIR / name
    writer  = ChunkWriter(out_dir, start_chunk, chunk_offset)

    lang_info = ""
    if cfg.get("quality_filter"):
        lang_info = f"  quality ≥ {cfg['quality_filter']}"
    if cfg.get("lang_filter"):
        lang_info += f"  lang={cfg['lang_filter']}"

    print(f"\n{'─'*60}")
    print(f"  Dataset : {name}  [Phase {cfg['phase']}]"
          + ("  [replay P3]" if cfg.get("replay") else ""))
    print(f"  Cible   : {token_target/1e9:.1f}B tokens  →  {out_dir}/" + lang_info)
    if doc_offset:
        print(f"  Reprise : doc #{doc_offset:,}  chunk #{start_chunk:03d}  "
              f"({tokens_done/1e9:.3f}B tokens déjà écrits)")
    print(f"{'─'*60}")

    t0   = time.time()
    pbar = tqdm(
        total      = token_target,
        initial    = tokens_done,
        unit       = "tok",
        unit_scale = True,
        desc       = name,
        dynamic_ncols=True,
    )

    doc_count  = doc_offset
    save_every = 10_000
    last_save  = doc_count
    t0_speed   = time.time()
    toks_since_save = 0

    for text in stream_dataset(cfg, doc_offset):
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(EOS_ID)
        writer.write(ids)
        tokens_done     += len(ids)
        toks_since_save += len(ids)
        doc_count       += 1

        pbar.update(len(ids))
        pbar.set_postfix(
            docs   = f"{doc_count:,}",
            chunks = f"{writer.total_chunks_closed}",
            speed  = f"{toks_since_save / max(time.time()-t0_speed, 1) / 1e6:.2f}M/s",
        )

        if tokens_done >= token_target:
            break

        if doc_count - last_save >= save_every:
            state.setdefault("in_progress", {})[name] = {
                "chunk_idx"  : writer.chunk_idx,
                "chunk_toks" : writer.chunk_toks,
                "doc_offset" : doc_count,
                "tokens_done": tokens_done,
            }
            save_resume(state)
            last_save       = doc_count
            toks_since_save = 0
            t0_speed        = time.time()

    pbar.close()
    final_chunk, final_toks = writer.close()

    n_chunks = final_chunk + (1 if final_toks > 0 else 0)
    print(f"  ✓ {name} : {tokens_done/1e9:.3f}B tokens  "
          f"{doc_count:,} docs  {n_chunks} chunks")

    state.setdefault("completed", []).append(name)
    state.get("in_progress", {}).pop(name, None)
    if name in state.get("failed", []):
        state["failed"].remove(name)
    save_resume(state)
    return tokens_done


# ── Scan d'un fichier chunk ────────────────────────────────────────────────────

def scan_chunk(path: Path) -> list[tuple[int, int]]:
    """Retourne [(start, length), ...] pour chaque document du chunk."""
    data          = np.fromfile(path, dtype=DTYPE)
    eos_positions = np.where(data == EOS_ID)[0]
    docs  = []
    start = 0
    for pos in eos_positions:
        length = int(pos) - start + 1
        if length > 1:
            docs.append((start, length))
        start = int(pos) + 1
    return docs


def scan_dataset_chunks(name: str) -> list[tuple[Path, int, int]]:
    """Scanne tous les chunks d'un dataset → [(chunk_path, offset, length), ...]"""
    out_dir   = CHUNKS_DIR / name
    chunks    = sorted(out_dir.glob("chunk_*.bin"))
    all_docs  : list[tuple[Path, int, int]] = []
    for chunk_path in tqdm(chunks, desc=f"  scan {name}", unit="chunk", leave=False):
        for start, length in scan_chunk(chunk_path):
            all_docs.append((chunk_path, start, length))
    return all_docs


# ── Écriture finale ────────────────────────────────────────────────────────────

def _read_doc(chunk_path: Path, offset: int, length: int) -> np.ndarray:
    # uint32 = 4 octets par token (correction : était * 2 pour uint16)
    with open(chunk_path, "rb") as f:
        f.seek(offset * 4)
        return np.frombuffer(f.read(length * 4), dtype=DTYPE).copy()


def _split_path(idx: int) -> Path:
    return Path(f"{OUT_PREFIX}_{idx:03d}.bin")


def upload_split_to_hf(file_path: Path, file_idx: int):
    """Upload un fichier pretrain_data_NNN.bin sur HuggingFace Hub."""
    if not HF_TOKEN:
        print(f"  ⚠  HF_TOKEN absent — upload ignoré pour {file_path.name}")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        print(f"  ↑ Upload {file_path.name} → {HF_DATASET_REPO} …")
        api.upload_file(
            path_or_fileobj = str(file_path),
            path_in_repo    = file_path.name,
            repo_id         = HF_DATASET_REPO,
            repo_type       = "dataset",
            commit_message  = f"pretrain split {file_idx:03d} ({file_path.stat().st_size/1e9:.1f} GB)",
        )
        print(f"  ✓ Upload OK → {HF_DATASET_REPO}/{file_path.name}")
    except ImportError:
        print("  ✗ huggingface_hub non installé : pip install huggingface_hub")
    except Exception as e:
        print(f"  ✗ Upload échoué pour {file_path.name} : {e}")


class _SplitWriter:
    """
    Écrit un flux de documents dans des fichiers de SPLIT_TOKENS tokens.
    Dès qu'un fichier atteint SPLIT_TOKENS tokens, il est fermé et uploadé sur HF.
    """

    def __init__(self, start_idx: int = 0):
        self.file_idx  = start_idx
        self.toks_in   = 0          # tokens écrits dans le fichier courant
        self.total     = 0          # tokens totaux écrits
        self.buf: list[np.ndarray] = []
        self.buf_size  = 0
        self._open()

    def _open(self):
        p = _split_path(self.file_idx)
        print(f"\n  ✎ Ouverture {p.name}  (fichier {self.file_idx+1})")
        self.f = open(p, "wb")

    def _flush(self):
        if self.buf:
            np.concatenate(self.buf).tofile(self.f)
            self.f.flush()
            self.buf      = []
            self.buf_size = 0

    def write_doc(self, arr: np.ndarray):
        remaining = arr
        while len(remaining):
            space = SPLIT_TOKENS - self.toks_in
            take  = min(space, len(remaining))
            self.buf.append(remaining[:take])
            self.buf_size  += take
            self.toks_in   += take
            self.total     += take
            remaining       = remaining[take:]

            if self.buf_size >= WRITE_BUF_TOKS:
                self._flush()

            if self.toks_in >= SPLIT_TOKENS:
                self._flush()
                self.f.close()
                p = _split_path(self.file_idx)
                print(f"  ✓ {p.name} complet : {self.toks_in/1e9:.1f}B tokens  "
                      f"({p.stat().st_size/1e9:.1f} GB)")
                upload_split_to_hf(p, self.file_idx)
                self.file_idx += 1
                self.toks_in  = 0
                self._open()

    def close(self) -> int:
        """Ferme le dernier fichier (peut être partiel) et l'uploade."""
        self._flush()
        self.f.close()
        p = _split_path(self.file_idx)
        if p.stat().st_size > 0:
            print(f"  ✓ {p.name} (dernier) : {self.toks_in/1e9:.1f}B tokens  "
                  f"({p.stat().st_size/1e9:.1f} GB)")
            upload_split_to_hf(p, self.file_idx)
            return self.file_idx + 1    # nombre total de fichiers
        else:
            p.unlink(missing_ok=True)
            return self.file_idx


def _write_docs(docs: list[tuple[Path, int, int]], writer: "_SplitWriter",
                desc: str = ""):
    pbar = tqdm(docs, desc=desc, unit="doc", dynamic_ncols=True, leave=False)
    for chunk_path, offset, length in pbar:
        arr = _read_doc(chunk_path, offset, length)
        writer.write_doc(arr)
    pbar.close()


def _interleave_replay(
    p2_docs    : list[tuple[Path, int, int]],
    replay_docs: list[tuple[Path, int, int]],
    ratio_start: float = REPLAY_RATIO_START,
    ratio_end  : float = REPLAY_RATIO_END,
    seed       : int   = 42,
) -> list[tuple[Path, int, int]]:
    """
    Interleave progressif des docs replay EN dans la Phase 3.
    Le ratio replay croît linéairement de ratio_start à ratio_end.
    """
    result  : list[tuple[Path, int, int]] = []
    r_idx   = 0
    budget  = 0.0
    n_p2    = len(p2_docs)
    n_r     = len(replay_docs)

    for i, doc in enumerate(p2_docs):
        frac   = i / max(n_p2 - 1, 1)
        ratio  = ratio_start + (ratio_end - ratio_start) * frac
        budget += ratio / max(1.0 - ratio, 1e-6)
        result.append(doc)

        while budget >= 1.0 and r_idx < n_r:
            result.append(replay_docs[r_idx])
            r_idx  += 1
            budget -= 1.0

    while r_idx < n_r:
        result.append(replay_docs[r_idx])
        r_idx += 1

    return result


# ── Assembly curriculum 3 phases ──────────────────────────────────────────────

def assemble(state: dict):
    print("\n" + "═" * 60)
    print("  ASSEMBLY — curriculum 3 phases  →  pretrain_data.bin")
    print("═" * 60)

    # ── Calcul du total de tokens tokenisés sur disque ───────────────────────
    total_tokens_on_disk = 0
    for d in DATASETS:
        out_dir = CHUNKS_DIR / d["name"]
        for p in out_dir.glob("chunk_*.bin"):
            total_tokens_on_disk += p.stat().st_size // 4    # uint32 = 4 octets

    p1_end_tok    = int(total_tokens_on_disk * P1_END_FRAC)
    p3_start_tok  = int(total_tokens_on_disk * P3_START_FRAC)
    print(f"\n  Tokens totaux sur disque : {total_tokens_on_disk/1e9:.1f}B")
    print(f"  Phase 1 : 0 → {p1_end_tok/1e9:.1f}B  ({P1_END_FRAC*100:.0f}%)")
    print(f"  Phase 2 : {p1_end_tok/1e9:.1f}B → {p3_start_tok/1e9:.1f}B  "
          f"({(P3_START_FRAC-P1_END_FRAC)*100:.0f}%)")
    print(f"  Phase 3 : {p3_start_tok/1e9:.1f}B → fin  "
          f"({(1-P3_START_FRAC)*100:.0f}%)")

    # ── 1. Scan Cosmopedia (Phase 1) ─────────────────────────────────────────
    print("\n[1/5] Scan CosmopediaV2…")
    cosmo_docs = scan_dataset_chunks("cosmopedia_v2")
    print(f"      {len(cosmo_docs):,} documents")

    cosmo_toks = sum(l for _, _, l in cosmo_docs)
    if cosmo_toks <= p1_end_tok:
        p1_docs       = cosmo_docs
        cosmo_p2_docs = []
    else:
        acc   = 0
        split = 0
        for i, (_, _, l) in enumerate(cosmo_docs):
            if acc + l > p1_end_tok:
                split = i
                break
            acc += l
        p1_docs       = cosmo_docs[:split]
        cosmo_p2_docs = cosmo_docs[split:]

    print(f"      → Phase 1 : {len(p1_docs):,} docs")
    print(f"      → Phase 2 : {len(cosmo_p2_docs):,} docs (excédent Cosmo)")

    # ── 2. Scan Phase 2 (tous datasets sauf Cosmo) ───────────────────────────
    print("\n[2/5] Scan datasets Phase 2…")
    p2_names = [d["name"] for d in DATASETS if d["name"] != "cosmopedia_v2"]
    all_p2_docs: list[tuple[Path, int, int]] = list(cosmo_p2_docs)

    for name in p2_names:
        ds_docs = scan_dataset_chunks(name)
        print(f"      {name:<30} {len(ds_docs):>8,} docs")
        all_p2_docs.extend(ds_docs)

    print(f"\n      Total Phase 2 brut : {len(all_p2_docs):,} documents")

    # ── 3. Shuffle Phase 2 ───────────────────────────────────────────────────
    print("\n[3/5] Shuffle Phase 2…")
    rng = random.Random(42)
    rng.shuffle(all_p2_docs)
    print(f"      ✓ {len(all_p2_docs):,} docs shufflés (seed=42)")

    p1_token_count = sum(l for _, _, l in p1_docs)
    acc   = p1_token_count
    split = len(all_p2_docs)
    for i, (_, _, l) in enumerate(all_p2_docs):
        if acc >= p3_start_tok:
            split = i
            break
        acc += l

    p2_docs      = all_p2_docs[:split]
    p3_base_docs = all_p2_docs[split:]

    print(f"      Phase 2 pure  : {len(p2_docs):,} docs")
    print(f"      Phase 3 base  : {len(p3_base_docs):,} docs")

    # ── 4. Docs replay pour Phase 3 ──────────────────────────────────────────
    print("\n[4/5] Préparation replay EN Phase 3…")

    p3_base_toks  = sum(l for _, _, l in p3_base_docs)
    ratio_moy     = (REPLAY_RATIO_START + REPLAY_RATIO_END) / 2
    replay_budget = int(p3_base_toks * ratio_moy / max(1.0 - ratio_moy, 1e-6))
    print(f"      Tokens Phase 3 base   : {p3_base_toks/1e9:.2f}B")
    print(f"      Budget replay cible   : {replay_budget/1e9:.2f}B tokens "
          f"(ratio moy {ratio_moy*100:.0f}%)")

    replay_docs: list[tuple[Path, int, int]] = []
    for name in REPLAY_DATASETS:
        rd = scan_dataset_chunks(name)
        rng.shuffle(rd)
        per_ds_budget = replay_budget // len(REPLAY_DATASETS)
        acc_r = 0
        taken = []
        for doc in rd:
            if acc_r >= per_ds_budget:
                break
            taken.append(doc)
            acc_r += doc[2]
        replay_docs.extend(taken)
        print(f"      {name:<30} {len(taken):>8,} docs  ({acc_r/1e9:.2f}B tokens)")

    rng.shuffle(replay_docs)
    p3_docs = _interleave_replay(p3_base_docs, replay_docs)
    print(f"      ✓ Phase 3 finale : {len(p3_docs):,} docs "
          f"({len(p3_base_docs):,} P2 + {len(replay_docs):,} replay)")

    # ── 5. Écriture en fichiers de 25B tokens ─────────────────────────────────
    total_docs = len(p1_docs) + len(p2_docs) + len(p3_docs)
    n_splits   = max(1, (sum(l for _,_,l in p1_docs+p2_docs+p3_docs)
                         + SPLIT_TOKENS - 1) // SPLIT_TOKENS)
    print(f"\n[5/5] Écriture → {OUT_PREFIX}_NNN.bin  "
          f"({total_docs:,} docs  ~{n_splits} fichiers de 25B tokens)")
    print(f"      Chaque fichier sera uploadé sur {HF_DATASET_REPO} dès sa clôture.")

    writer = _SplitWriter(start_idx=0)

    print(f"\n  ━━ Phase 1 : Cosmopedia pur ({len(p1_docs):,} docs) ━━")
    _write_docs(p1_docs, writer, desc="  Phase 1")

    print(f"\n  ━━ Phase 2 : Mix multilingue shufflé ({len(p2_docs):,} docs) ━━")
    _write_docs(p2_docs, writer, desc="  Phase 2")

    print(f"\n  ━━ Phase 3 : Replay EN progressif ({len(p3_docs):,} docs) ━━")
    _write_docs(p3_docs, writer, desc="  Phase 3")

    n_files = writer.close()
    print(f"\n  ✓ Assembly terminé : {writer.total/1e9:.3f}B tokens  "
          f"{n_files} fichier(s)  ({writer.total*4/1e9:.0f} GB total)")


# ── Résumé ─────────────────────────────────────────────────────────────────────

def print_summary():
    total_target = sum(d["token_target"] for d in DATASETS)
    print(f"\n{'━'*60}")
    print(f"  RÉSUMÉ — mix 200B tokens")
    print(f"{'━'*60}")
    print(f"  {'Nom':<28} {'Ph':>2}  {'Cible':>6}  {'Disque':>8}  {'Chunks'}")
    print(f"  {'─'*56}")
    for d in DATASETS:
        out_dir = CHUNKS_DIR / d["name"]
        chunks  = list(out_dir.glob("chunk_*.bin")) if out_dir.exists() else []
        actual  = sum(p.stat().st_size for p in chunks) // 4 if chunks else 0
        replay  = " ♻" if d.get("replay") else ""
        print(f"  {d['name']:<28} {d['phase']:>2}  "
              f"{d['token_target']/1e9:>5.1f}B  "
              f"{actual/1e9:>7.2f}B  "
              f"{len(chunks)} chunks{replay}")
    print(f"  {'─'*56}")
    print(f"  {'TOTAL':<28}     {total_target/1e9:>5.1f}B")
    print(f"{'━'*60}")
    print(f"\n  Curriculum :")
    print(f"    Phase 1 ({P1_END_FRAC*100:.0f}%)  : Cosmopedia pur")
    print(f"    Phase 2 ({(P3_START_FRAC-P1_END_FRAC)*100:.0f}%)  : Mix EN+FR+JA+Code shufflé")
    print(f"    Phase 3 ({(1-P3_START_FRAC)*100:.0f}%)  : Replay EN HQ progressif "
          f"({REPLAY_RATIO_START*100:.0f}%→{REPLAY_RATIO_END*100:.0f}%)")
    print(f"  Replay datasets : {', '.join(REPLAY_DATASETS)}")
    print(f"{'━'*60}")
    splits = sorted(Path(".").glob(f"{OUT_PREFIX}_*.bin"))
    if splits:
        total_split_toks = sum(p.stat().st_size for p in splits) // 4
        total_split_gb   = sum(p.stat().st_size for p in splits) / 1e9
        print(f"\n  Fichiers de pretrain ({len(splits)}) :")
        for p in splits:
            n = p.stat().st_size // 4
            print(f"    {p.name}  {n/1e9:.1f}B tokens  ({p.stat().st_size/1e9:.1f} GB)")
        print(f"  Total : {total_split_toks/1e9:.1f}B tokens  ({total_split_gb:.0f} GB)")
    print(f"{'━'*60}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tokenisation + assembly curriculum 200B tokens (EN/FR/JA/Code)"
    )
    parser.add_argument("--reset",         action="store_true",
                        help="Repart de zéro (supprime resume.json)")
    parser.add_argument("--skip-assembly", action="store_true",
                        help="Tokenise uniquement, sans assembly final")
    parser.add_argument("--only-assembly", action="store_true",
                        help="Saute le download, lance seulement l'assembly")
    parser.add_argument("--no-preflight",  action="store_true",
                        help="Saute le pre-flight check (déconseillé)")
    args = parser.parse_args()

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        if RESUME_FILE.exists():
            RESUME_FILE.unlink()
            print("⚠  Fichier de reprise supprimé — repartir de zéro.")

    # ── Pre-flight ───────────────────────────────────────────────────────────
    if not args.only_assembly and not args.no_preflight:
        ok = preflight_check(DATASETS)
        if not ok:
            sys.exit(1)

    state = (load_resume() if not args.reset
             else {"completed": [], "in_progress": {}, "failed": []})

    # ── Download + tokenisation ──────────────────────────────────────────────
    if not args.only_assembly:
        tokenizer = load_tokenizer()
        random.seed(42)

        completed   = state.get("completed", [])
        prev_failed = state.get("failed", [])
        pending     = [d for d in DATASETS if d["name"] not in completed]

        if prev_failed:
            print(f"\n  ⚠  {len(prev_failed)} dataset(s) avaient échoué :")
            for n in prev_failed:
                print(f"     - {n}")

        for cfg in pending:
            try:
                download_dataset(cfg, tokenizer, state)
            except Exception as exc:
                name = cfg["name"]
                print(f"\n  ✗ {name} a échoué : {exc}")
                print(f"    → marqué 'failed' — sera retesté au prochain lancement")
                failed_list = state.setdefault("failed", [])
                if name not in failed_list:
                    failed_list.append(name)
                state.get("in_progress", {}).pop(name, None)
                save_resume(state)

        n_failed = len(state.get("failed", []))
        if n_failed:
            print(f"\n  ⚠  {n_failed} dataset(s) ont échoué.")
            print(f"     Relance : python tokens.py")
        else:
            print("\n  ✓ Tokenisation terminée.")

    # ── Assembly curriculum ──────────────────────────────────────────────────
    if not args.skip_assembly:
        assemble(state)

    print_summary()
    print("\n✓ Terminé.")


if __name__ == "__main__":
    main()
