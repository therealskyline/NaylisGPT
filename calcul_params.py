import sys
import math
import argparse
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib


def _ffn_hidden(embed_dim: int, multiple_of: int = 64) -> int:
    raw = int(embed_dim * 8 / 3)
    return ((raw + multiple_of - 1) // multiple_of) * multiple_of


def _gdn_block_params(embed_dim: int, gdn_head_dim: int, gdn_v_heads: int = 0) -> int:
    qk_heads   = embed_dim // gdn_head_dim
    qk_dim     = qk_heads * gdn_head_dim
    v_heads    = gdn_v_heads if gdn_v_heads else qk_heads
    v_dim      = v_heads * gdn_head_dim
    attn_params = (
        2 * embed_dim * qk_dim +
        2 * embed_dim * v_dim  +
        v_dim * embed_dim
    )
    ffn_hidden  = _ffn_hidden(embed_dim)
    ffn_params  = 2 * embed_dim * ffn_hidden + ffn_hidden * embed_dim
    norm_params = 2 * embed_dim
    beta_params = qk_heads
    return attn_params + ffn_params + norm_params + beta_params


def _gpt_block_params(
    embed_dim: int,
    num_heads: int,
    n_kv_heads: int,
    use_moe: bool,
    num_experts: int,
    top_k_experts: int,
    shared_experts: int,
    expert_hidden_dim: int,
    attn_head_dim: int = 0,
) -> dict:
    head_dim = attn_head_dim if attn_head_dim else (embed_dim // num_heads)
    qo_dim   = num_heads * head_dim
    kv_dim   = n_kv_heads * head_dim
    attn     = embed_dim * qo_dim + 2 * embed_dim * kv_dim + qo_dim * embed_dim
    norm     = 2 * embed_dim

    if use_moe:
        per_expert = 3 * embed_dim * expert_hidden_dim
        router     = embed_dim * num_experts
        ffn_total  = (num_experts + shared_experts) * per_expert + router
        ffn_active = (top_k_experts + shared_experts) * per_expert
    else:
        ffn_hidden = _ffn_hidden(embed_dim)
        ffn_total  = 3 * embed_dim * ffn_hidden
        ffn_active = ffn_total

    return {
        "attn":       attn,
        "ffn_total":  ffn_total,
        "ffn_active": ffn_active,
        "norm":       norm,
        "total":      attn + ffn_total + norm,
        "active":     attn + ffn_active + norm,
    }


def _mtp_module_params(embed_dim: int, vocab_size: int) -> int:
    proj   = embed_dim * 2 * embed_dim + embed_dim
    norms  = 3 * embed_dim
    head   = embed_dim * vocab_size
    return proj + norms + head


def compute(cfg: dict) -> dict:
    vocab_size     = cfg.get("vocab_size", 128256)
    embed_dim      = cfg["embed_dim"]
    num_heads      = cfg["num_heads"]
    num_layers     = cfg["num_layers"]
    n_kv_heads     = cfg.get("n_kv_heads", num_heads)
    hybrid_ratio   = cfg.get("hybrid_ratio", 3)
    gdn_head_dim   = cfg.get("gdn_head_dim", embed_dim // num_heads)

    use_moe           = cfg.get("use_moe", False)
    num_experts       = cfg.get("num_experts", 8)
    top_k_experts     = cfg.get("top_k_experts", 2)
    shared_experts    = cfg.get("shared_experts", 2)
    expert_hidden_dim = cfg.get("expert_hidden_dim") or _ffn_hidden(embed_dim)
    mtp_steps         = cfg.get("mtp_steps", 0)

    num_gpt = num_layers // (hybrid_ratio + 1)
    num_gdn = num_layers - num_gpt

    emb_params  = vocab_size * embed_dim
    ln_params   = embed_dim

    gdn_v_heads   = cfg.get("gdn_v_heads", 0)
    attn_head_dim = cfg.get("attn_head_dim", 0)

    gdn_per     = _gdn_block_params(embed_dim, gdn_head_dim, gdn_v_heads)
    gdn_total   = num_gdn * gdn_per

    gpt_per     = _gpt_block_params(
        embed_dim, num_heads, n_kv_heads,
        use_moe, num_experts, top_k_experts, shared_experts, expert_hidden_dim,
        attn_head_dim=attn_head_dim,
    )
    gpt_total   = num_gpt * gpt_per["total"]
    gpt_active  = num_gpt * gpt_per["active"]

    mtp_params  = 0
    if mtp_steps > 0:
        mtp_params = mtp_steps * _mtp_module_params(embed_dim, vocab_size)

    total_params  = emb_params + gdn_total + gpt_total   + ln_params + mtp_params
    active_params = emb_params + gdn_total + gpt_active  + ln_params

    _h           = attn_head_dim if attn_head_dim else (embed_dim // num_heads)
    ffn_hidden_d = _ffn_hidden(embed_dim)
    ffn_hidden_e = expert_hidden_dim if use_moe else ffn_hidden_d

    return {
        "embed_dim":     embed_dim,
        "num_heads":     num_heads,
        "head_dim":      _h,
        "n_kv_heads":    n_kv_heads,
        "gqa_ratio":     num_heads // n_kv_heads,
        "num_layers":    num_layers,
        "num_gdn":       num_gdn,
        "num_gpt":       num_gpt,
        "hybrid_ratio":  hybrid_ratio,
        "gdn_head_dim":  gdn_head_dim,
        "ffn_hidden":    ffn_hidden_d,
        "use_moe":       use_moe,
        "num_experts":   num_experts,
        "shared_exp":    shared_experts,
        "top_k_experts": top_k_experts,
        "expert_hidden": ffn_hidden_e,
        "mtp_steps":     mtp_steps,
        "vocab_size":    vocab_size,
        "emb_params":    emb_params,
        "gdn_per":       gdn_per,
        "gdn_total":     gdn_total,
        "gpt_per":       gpt_per,
        "gpt_total":     gpt_total,
        "gpt_active":    gpt_active,
        "mtp_params":    mtp_params,
        "total_params":  total_params,
        "active_params": active_params,
    }


def _fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.3f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def _bar(frac: float, width: int = 30) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def print_report(name: str, r: dict):
    sep  = "─" * 60
    sep2 = "═" * 60

    print(f"\n{sep2}")
    print(f"  Config : {name}")
    print(sep2)

    print(f"\n  Architecture")
    print(f"  {'embed_dim':<26} {r['embed_dim']}")
    print(f"  {'num_layers':<26} {r['num_layers']}  ({r['num_gdn']} GDN + {r['num_gpt']} GPT, ratio {r['hybrid_ratio']}:1)")
    print(f"  {'num_heads (GPT full)':<26} {r['num_heads']}  (head_dim={r['head_dim']})")
    print(f"  {'n_kv_heads (GQA)':<26} {r['n_kv_heads']}  → {r['gqa_ratio']}:1 ratio")
    print(f"  {'gdn_head_dim':<26} {r['gdn_head_dim']}")
    print(f"  {'ffn_hidden (dense base)':<26} {r['ffn_hidden']}")
    print(f"  {'vocab_size':<26} {r['vocab_size']:,}")

    if r["use_moe"]:
        print(f"\n  MoE  ({r['num_experts']} routed + {r['shared_exp']} shared, top-{r['top_k_experts']})")
        print(f"  {'expert_hidden_dim':<26} {r['expert_hidden']}")
        print(f"  {'experts actifs/token':<26} {r['top_k_experts']} routés + {r['shared_exp']} shared")

    if r["mtp_steps"] > 0:
        print(f"\n  MTP  {r['mtp_steps']} étape(s)")

    print(f"\n{sep}")
    print(f"  {'Composant':<32} {'Total':>10}  {'Actif':>10}  {'%':>6}")
    print(sep)

    total = r["total_params"]

    rows = [
        ("Embeddings  (token)",  r["emb_params"],   r["emb_params"]),
        (f"GDN blocks  ({r['num_gdn']} × {_fmt(r['gdn_per'])} ea.)", r["gdn_total"], r["gdn_total"]),
        (f"GPT blocks  ({r['num_gpt']} × ~{_fmt(r['gpt_per']['total'])} ea.)", r["gpt_total"], r["gpt_active"]),
        ("MTP modules",          r["mtp_params"],   r["mtp_params"]),
        ("LN final",             r.get("ln_params", r["embed_dim"]), r.get("ln_params", r["embed_dim"])),
    ]

    for label, tot, act in rows:
        if tot == 0:
            continue
        pct = tot / total * 100
        print(f"  {label:<32} {_fmt(tot):>10}  {_fmt(act):>10}  {pct:>5.1f}%")

    print(sep)
    print(f"  {'TOTAL':.<32} {_fmt(r['total_params']):>10}  {_fmt(r['active_params']):>10}")

    if r["use_moe"]:
        eff = r["active_params"] / r["total_params"] * 100
        print(f"\n  Efficacité MoE : {eff:.1f}% des params actifs par token")
        print(f"  {_bar(eff/100)}  {eff:.1f}%")

    print(f"\n  Mémoire estimée  (BF16 = 2 o/param)")
    for label, mult, note in [
        ("Poids seuls",      2, "inférence min"),
        ("Poids + gradients",4, "entraînement min"),
        ("Poids + grads + AdamW", 12, "entraînement complet"),
    ]:
        mem_gb = r["total_params"] * mult / 1024**3
        print(f"    {label:<30} {mem_gb:>6.1f} Go  ({note})")

    print()


def compare(configs: list[dict]):
    names = [c["_name"] for c in configs]
    totals = [c["total_params"] for c in configs]
    actives = [c["active_params"] for c in configs]

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  Comparatif  {len(configs)} configs")
    print(sep)
    print(f"  {'Config':<15} {'Embed':>7} {'Layers':>7} {'Total':>10} {'Actif':>10} {'Efficacité':>12}")
    print(sep)
    for c in configs:
        eff = c["active_params"]/c["total_params"]*100 if c["total_params"] else 0
        eff_str = f"{eff:.0f}%" if c["use_moe"] else "dense"
        print(
            f"  {c['_name']:<15}"
            f" {c['embed_dim']:>7}"
            f" {c['num_layers']:>7}"
            f" {_fmt(c['total_params']):>10}"
            f" {_fmt(c['active_params']):>10}"
            f" {eff_str:>12}"
        )
    print(sep)
    print()


def _load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    flat: dict = {}
    for section, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[section] = value

    if "soft_cap" in flat and flat["soft_cap"] == 0:
        flat["soft_cap"] = None

    return flat


def main():
    parser = argparse.ArgumentParser(description="Calcule les paramètres d'un config NaylisGDN.")
    parser.add_argument("configs", nargs="*", help="Fichiers TOML (défaut: tous dans config/)")
    parser.add_argument("--compare", action="store_true", help="Affiche tableau comparatif")
    args = parser.parse_args()

    config_dir = Path(__file__).parent / "config"

    if args.configs:
        paths = [Path(p) for p in args.configs]
    else:
        paths = sorted(config_dir.glob("*.toml"))

    if not paths:
        print("Aucun fichier TOML trouvé.")
        sys.exit(1)

    results = []
    for p in paths:
        cfg  = _load_toml(str(p))
        name = p.stem
        r    = compute(cfg)
        r["_name"] = name
        results.append(r)
        print_report(name, r)

    if args.compare or len(results) > 1:
        compare(results)


if __name__ == "__main__":
    main()
