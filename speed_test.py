import torch
import torch.nn.functional as F
import math
import time
import gc
import sys
import argparse
import traceback
from typing import Optional

parser = argparse.ArgumentParser(description="NaylisGDN Speed Benchmark")
parser.add_argument("--batch-size",   type=int,   default=28)
parser.add_argument("--seq-len",      type=int,   default=512)
parser.add_argument("--embed-dim",    type=int,   default=1280)
parser.add_argument("--num-heads",    type=int,   default=20)
parser.add_argument("--num-layers",   type=int,   default=24)
parser.add_argument("--n-kv-heads",   type=int,   default=5)
parser.add_argument("--vocab-size",   type=int,   default=128256)
parser.add_argument("--soft-cap",     type=float, default=None)
parser.add_argument("--no-compile",   action="store_true")
parser.add_argument("--warmup-steps", type=int,   default=5)
parser.add_argument("--bench-steps",  type=int,   default=20)
args = parser.parse_args()

RESET  = "\033[0m";  BOLD   = "\033[1m"
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"

def ok(msg):     print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):   print(f"  {YELLOW}⚠️  {msg}{RESET}")
def bad(msg):    print(f"  {RED}❌ {msg}{RESET}")
def info(msg):   print(f"  {CYAN}ℹ️  {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'═'*70}\n  {msg}\n{'═'*70}{RESET}")

results = {}
device  = "cuda" if torch.cuda.is_available() else "cpu"
dtype   = torch.bfloat16

print(f"\n{BOLD}NaylisGDN Speed Benchmark{RESET}")
print(f"  batch={args.batch_size}  seq={args.seq_len}  embed={args.embed_dim}  "
      f"layers={args.num_layers}  heads={args.num_heads}  kv={args.n_kv_heads}")
print(f"  compile={'off' if args.no_compile else 'on'}")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, warmup=args.warmup_steps, steps=args.bench_steps):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    sync()
    return (time.perf_counter() - t0) / steps * 1000


header("TEST 1 — GPU Info & Capacité Théorique")

if device == "cpu":
    bad("Pas de GPU détecté")
    results["gpu"] = "cpu"
else:
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1e9
    print(f"  GPU       : {props.name}")
    print(f"  VRAM      : {vram:.0f} GB  |  SM : {props.multi_processor_count}")
    print(f"  CUDA caps : {props.major}.{props.minor}  |  PyTorch : {torch.__version__}")

    if props.major >= 9:
        ok(f"SM{props.major}{props.minor} — FP8 compatible (Hopper+)")
    else:
        bad(f"SM{props.major}{props.minor} — FP8 nécessite SM90+")

    tflops_map = {"B200": 1979, "H200": 1979, "H100": 1979, "A100": 312, "4090": 165}
    tflops     = next((v for k, v in tflops_map.items() if k in props.name), None)

    params = (
        args.vocab_size * args.embed_dim
        + args.num_layers * (
            args.embed_dim * args.num_heads * (args.embed_dim // args.num_heads) * 3
            + args.embed_dim * args.embed_dim
            + args.embed_dim * int(8 * args.embed_dim / 3 / 64) * 64 * 3
        )
    )
    print(f"\n  Params estimés : {params/1e9:.2f}B")

    if tflops:
        tokens_per_sec_theory = (tflops * 1e12) / (6 * params)
        print(f"  TFLOPs FP8 : {tflops} TF/s  |  100% MFU → {tokens_per_sec_theory/1e6:.0f}M tok/s")
        print(f"  Réaliste 50% MFU : {tokens_per_sec_theory*0.5/1e6:.0f}M tok/s")
        results["tflops_theory"] = tflops
    else:
        warn(f"GPU '{props.name}' inconnu dans la table TFLOPs")

    results.update({"gpu_name": props.name, "vram_gb": vram})


header("TEST 2 — FP8 Transformer Engine disponibilité")

try:
    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as _te_recipe

    fp8_recipe = _te_recipe.DelayedScaling(
        margin=0, fp8_format=_te_recipe.Format.HYBRID,
        amax_history_len=16, amax_compute_algo="max",
    )
    ok(f"Transformer Engine {transformer_engine.__version__} — FP8 DelayedScaling HYBRID")
    _TE_OK = True
except ImportError:
    bad("transformer_engine non installé — FP8 indisponible")
    _TE_OK = False
    fp8_recipe = None

try:
    F.scaled_dot_product_attention
    ok("F.scaled_dot_product_attention disponible")
    results["sdpa"] = True
except AttributeError:
    bad("F.scaled_dot_product_attention ABSENT (PyTorch < 2.0)")
    results["sdpa"] = False


header("TEST 3 — te.Linear FP8 vs nn.Linear BF16")

if _TE_OK and device == "cuda":
    import contextlib
    D      = args.embed_dim
    B, S   = args.batch_size, args.seq_len
    x_test = torch.randn(B, S, D, device=device, dtype=dtype)

    lin_bf16 = torch.nn.Linear(D, D * 4, bias=False).to(device).to(dtype)
    lin_fp8  = te.Linear(D, D * 4, bias=False).to(device).to(dtype)

    ms_bf16 = timeit(lambda: lin_bf16(x_test))
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        ms_fp8 = timeit(lambda: lin_fp8(x_test))

    speedup = ms_bf16 / ms_fp8
    color   = GREEN if speedup > 1.3 else (YELLOW if speedup > 1.0 else RED)
    ok(f"BF16 nn.Linear : {ms_bf16:.3f} ms")
    print(f"  {color}FP8  te.Linear : {ms_fp8:.3f} ms  ({speedup:.2f}x){RESET}")
    results["ms_bf16_linear"] = ms_bf16
    results["ms_fp8_linear"]  = ms_fp8
    del lin_bf16, lin_fp8
else:
    warn("Skip — TE non disponible ou CPU")


try:
    from naylisgdn import NaylisGDN
    MODEL_AVAILABLE = True
except ImportError as e:
    warn(f"NaylisGDN non importable ({e}) — tests modèle skippés")
    MODEL_AVAILABLE = False

x_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device) \
        if device == "cuda" else None
y_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device) \
        if device == "cuda" else None


def make_model(use_fp8=False):
    return NaylisGDN(
        vocab_size=args.vocab_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        max_seq_len=args.seq_len, dropout=0.0,
        use_rope=True, use_swiglu=True, n_kv_heads=args.n_kv_heads,
        use_qk_norm=True, soft_cap=args.soft_cap, use_flash_attn=True,
        use_fp8=use_fp8,
    ).to(device).to(dtype)

def _fp8_ctx_fn(use_fp8):
    if use_fp8 and _TE_OK and fp8_recipe is not None:
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    import contextlib
    return contextlib.nullcontext()


header("TEST 4 — Forward BF16 vs FP8")
if MODEL_AVAILABLE and device == "cuda":
    m_bf16 = make_model(use_fp8=False); m_bf16.eval()
    print(f"\n  {m_bf16.count_parameters()['total']/1e6:.1f}M paramètres")

    with torch.no_grad():
        ms_bf16 = timeit(lambda: m_bf16(x_ids))
        ok(f"Forward BF16 : {ms_bf16:.1f} ms  →  {args.batch_size*args.seq_len/(ms_bf16/1000)/1e6:.0f}M tok/s")

    del m_bf16; gc.collect(); torch.cuda.empty_cache()

    if _TE_OK:
        m_fp8 = make_model(use_fp8=True); m_fp8.eval()
        with torch.no_grad():
            with _fp8_ctx_fn(True):
                ms_fp8 = timeit(lambda: m_fp8(x_ids))
        speedup = ms_bf16 / ms_fp8
        color   = GREEN if speedup > 1.3 else (YELLOW if speedup > 1.0 else RED)
        print(f"  {color}Forward FP8  : {ms_fp8:.1f} ms  →  {args.batch_size*args.seq_len/(ms_fp8/1000)/1e6:.0f}M tok/s  ({speedup:.2f}x){RESET}")
        results.update({"ms_fwd_bf16": ms_bf16, "ms_fwd_fp8": ms_fp8, "speedup_fwd": speedup})
        del m_fp8; gc.collect(); torch.cuda.empty_cache()
else:
    warn("Skip")


header("TEST 5 — Forward + Backward + Optimizer (Muon + AdamW)")
if MODEL_AVAILABLE and device == "cuda":
    from naylisgdn.optimizers import configure_optimizers
    for use_fp8 in ([False, True] if _TE_OK else [False]):
        m = make_model(use_fp8=use_fp8); m.train()
        muon, adamw = configure_optimizers(m, lr=4e-4, weight_decay=0.1,
                                           betas=(0.9, 0.95), eps=1e-8, device=device)
        def full_step():
            with _fp8_ctx_fn(use_fp8):
                with torch.amp.autocast(device, dtype=dtype):
                    _, loss, _ = m(x_ids, targets=y_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            muon.step(); adamw.step()
            muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        ms = timeit(full_step, warmup=3, steps=10)
        label = "FP8" if use_fp8 else "BF16"
        ok(f"Full step {label} : {ms:.1f} ms/step  →  {args.batch_size*args.seq_len/(ms/1000)/1e6:.0f}M tok/s")
        results[f"ms_full_{label.lower()}"] = ms
        del m, muon, adamw; gc.collect(); torch.cuda.empty_cache()
else:
    warn("Skip")


header("TEST 6 — torch.compile")
if MODEL_AVAILABLE and device == "cuda" and not args.no_compile:
    m = make_model(use_fp8=False); m.eval()
    ms_before = timeit(lambda: m(x_ids), warmup=5, steps=20)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    try:
        m_c = torch.compile(m, mode="default")
        for _ in range(3):
            with torch.no_grad():
                m_c(x_ids)
        ms_after = timeit(lambda: m_c(x_ids), warmup=3, steps=20)
        speedup  = ms_before / ms_after
        color    = GREEN if speedup > 1.3 else (YELLOW if speedup > 1.1 else RED)
        ok(f"compile : {ms_before:.1f} ms → {ms_after:.1f} ms  ({color}{speedup:.2f}x{RESET})")
        results["compile_speedup"] = speedup
    except Exception as e:
        warn(f"torch.compile échoue : {e}")
    del m; gc.collect(); torch.cuda.empty_cache()
else:
    warn("Skip (--no-compile ou modèle non dispo)")


header("TEST 7 — Résumé")

if results.get("ms_full_fp8") and results.get("tflops_theory"):
    tok_s = args.batch_size * args.seq_len / (results["ms_full_fp8"] / 1000)
    mfu   = tok_s * 6 * (args.num_layers * args.embed_dim ** 2 * 12) / \
            (results["tflops_theory"] * 1e12) * 100
    ok(f"Throughput FP8 réel : {tok_s/1e6:.0f}M tok/s")
    ok(f"MFU FP8 estimé      : {mfu:.1f}%")

if results.get("speedup_fwd"):
    s = results["speedup_fwd"]
    color = GREEN if s > 1.3 else (YELLOW if s > 1.0 else RED)
    print(f"  {color}Speedup FP8 vs BF16 : {s:.2f}x{RESET}")

if results.get("compile_speedup", 1.0) > 1.3:
    ok(f"torch.compile utile : {results['compile_speedup']:.2f}x")

print()
