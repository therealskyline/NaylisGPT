# NaylisGDN

Modèle de langage hybride **Gated DeltaNet-2 + Transformer** (ratio 3:1), optimisé pour l'entraînement sur NVIDIA H100/B200. Trois tailles : 480M, 1B, 4B paramètres.

---

## Architecture

```
NaylisGDN = [GDNBlock, GDNBlock, GDNBlock, TransformerBlock] × N
```

Chaque groupe de 4 couches contient 3 blocs **GDN-2** (récurrent linéaire) et 1 bloc **GPT** (attention softmax), avec une tête MTP optionnelle pour la prédiction multi-token.

### Gated DeltaNet-2 (arXiv 2605.22791 — Hatamizadeh et al. 2026)

```
S_t = α_t * S_{t-1}
    - k_t ⊗ ((b_t * k_t)ᵀ S_{t-1}_decay)    # EFFACER  (erase gate b_t)
    + k_t ⊗ (w_t * v_t)                      # ÉCRIRE   (write gate w_t)
y_t = S_t q_t
```

### Attention (blocs GPT, ratio 1/4)

- GQA — `n_kv_heads` < `num_heads` selon config
- RoPE avec `rope_base=1_000_000` (style Qwen3), `rope_dim` configurable
- Flash Attention 2/3/4
- QK-Norm activé (`use_qk_norm = true`)

---

## Configurations

| Config | Params | embed_dim | Couches | Têtes attn | n_kv | attn_head_dim | Tokenizer | vocab_size |
|--------|--------|-----------|---------|------------|------|---------------|-----------|-----------|
| 480M.toml | ~480M | 1280 | 24 | 20 | 5 | — | cosmo2 | 49 152 |
| 1B.toml | ~1B | 1408 | 32 | 8 | 2 | 256 | cosmo2 | 49 152 |
| 4B.toml | ~4B | 2560 | 32 | 16 | 4 | 256 | Qwen3 | 151 936 |

GDN config 1B : `gdn_head_dim=128`, `gdn_heads_v=16`, `gdn_heads_qk=16`  
Ratio GDN-2/GPT = 3:1 dans toutes les configs.  
`special_tokens = []` pour toutes les configs (cosmo2 et Qwen3 n'ont rien à ajouter).

---

## Stack

| Composant | Technologie |
|-----------|-------------|
| Précision | FP8 `DelayedScaling HYBRID` (SM90+) via Transformer Engine, sinon BF16 |
| GDN-2 kernel | `fla.ops.gated_delta_rule_2` (Triton) → fallback PyTorch |
| Norms | `LigerRMSNorm` (Triton fusionné) → fallback PyTorch |
| Loss | `LigerFusedLinearCrossEntropyLoss` |
| Optimiseur | Muon + AdamW |
| Scheduler | WSD (Warmup-Stable-Decay) |
| Données | Sequence packing, memmap uint32 |
| Multi-GPU | FSDP `FULL_SHARD` (auto-détecté via `torchrun`) |
| Compilation | `torch.compile` + `coordinate_descent_tuning` |

---

## Pipeline de données (50B tokens)

```bash
python data/tokens.py                  # full run — download + assembly
python data/tokens.py --skip-assembly  # download seulement
python data/tokens.py --only-assembly  # assembly seulement
python data/tokens.py --reset          # repart de zéro
```

### Dataset mix (config 1B)

| Dataset | Phase | Tokens |
|---------|-------|--------|
| HuggingFaceTB/smollm-corpus [cosmopedia-v2] | 1 | 15B |
| nv-community/Nemotron-CC-v2.1 [Non-Synth HQ] | 2 | 10B |
| tokyotech-llm/swallow-code-v2 [Python, no-JP] | 2 | 7B |
| HuggingFaceFW/finephrase [all] | 2 | 7B |
| nv-community/Nemotron-CC-Math-v1 [4plus] | 2 | 6B |
| nv-community/Nemotron-CC-v2.1 [High-Synth] | 2 | 5B |
| **Total** | | **50B** |

- **Phase 1** (0 → 30%) : Cosmopedia pur → base syntaxique anglaise
- **Phase 2** (30 → 100%) : Tout shufflé → généralisation multi-domaine
- SwallowCode filtré Python uniquement + `skip_jp=True` (zéro kanji/kana)

### Output : 5 fichiers × 10B tokens (~40 GB chacun)

```
pretrain_data_000.bin  … pretrain_data_004.bin
```

Chaque fichier uploadé sur HuggingFace Hub dès sa clôture.

---

## Entraînement

```bash
# Single GPU
python train.py --config config/1B.toml --hf-token <token> --hf-repo <repo>

# Multi-GPU
torchrun --nproc_per_node=8 train.py --config config/4B.toml --hf-token <token>
```

### Flow chunk par chunk

```
pretrain_data_000.bin  →  chunk 1/5  (~1 200 steps)  →  checkpoint HF
pretrain_data_001.bin  →  chunk 2/5  (~1 200 steps)  →  checkpoint HF
pretrain_data_002.bin  →  chunk 3/5  (~1 200 steps)  →  checkpoint HF
pretrain_data_003.bin  →  chunk 4/5  (~1 200 steps)  →  checkpoint HF
pretrain_data_004.bin  →  chunk 5/5  (~1 200 steps)  →  checkpoint HF
```

- **Save local** : toutes les 1H (`save_every_hours = 1.0`)
- **Push HF** : à chaque save local (`hf_push_interval = 1800`)
- **Reprise** : `actual_chunk_done` + `skip_batches` dans le checkpoint JSON → reprise exacte mid-chunk

---

## Précision FP8

`DelayedScaling(margin=0, fp8_format=HYBRID, amax_history_len=16)` :

- **E4M3** forward + **E5M2** backward
- Activé si `use_fp8 = true` **et** GPU SM90+ (H100/H200) détecté
- Fallback BF16 automatique sinon

---

## Structure

```
NaylisGDN/
├── naylisgdn/
│   ├── model.py              — NaylisGDN, MTP, comptage params
│   ├── gdn_block.py          — Gated DeltaNet-2 (arXiv 2605.22791)
│   ├── attention.py          — MHA + GQA + Flash Attention
│   ├── transformer_block.py
│   ├── feedforward.py        — SwiGLU / te.Linear FP8
│   ├── norm.py               — RMSNorm (Liger Triton ou PyTorch)
│   ├── rope.py               — RoPE / YaRN
│   ├── optimizers.py         — Muon + AdamW
│   └── scheduler.py          — WSDScheduler
├── config/
│   ├── 480M.toml             — ~480M params, cosmo2, vocab 49 152
│   ├── 1B.toml               — ~1B params,  cosmo2, vocab 49 152, seq 8192
│   └── 4B.toml               — ~4B params,  Qwen3,  vocab 151 936
├── data/
│   ├── tokens.py             — download 50B tokens → 5×10B .bin + upload HF
│   └── translate_and_tokenize.py  — pipeline JP→EN vLLM (SwallowCode, optionnel)
├── train.py                  — boucle d'entraînement (FP8, FSDP, compile, reprise)
└── requirements.txt
```
