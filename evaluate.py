import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from naylisgdn import NaylisGDN

try:
    from lm_eval.api.model import LM
    from lm_eval import simple_evaluate
except ImportError:
    sys.exit(
        "lm-evaluation-harness non trouvé.\n"
        "Installe-le avec : pip install lm-eval==0.4.3"
    )

_ARCH_KEYS = [
    "vocab_size", "embed_dim", "num_heads", "num_layers", "max_seq_len",
    "dropout", "use_rope", "use_yarn", "yarn_scale", "yarn_original_max_len",
    "use_swiglu", "n_kv_heads", "use_qk_norm", "soft_cap", "use_flash_attn",
    "hybrid_ratio", "gdn_head_dim",
]

TOKENIZER_ID       = "HuggingFaceTB/cosmo2-tokenizer"
DEFAULT_MODEL_PATH = "./Model/NaylisGDN_pretrain.pt"

MODEL_CFG = dict(
    vocab_size            = None,
    embed_dim             = 1280,
    num_heads             = 20,
    num_layers            = 24,
    max_seq_len           = 1024,
    n_kv_heads            = 5,
    use_rope              = True,
    use_yarn              = False,
    yarn_scale            = 4.0,
    yarn_original_max_len = 512,
    use_swiglu            = True,
    use_qk_norm           = True,
    soft_cap              = None,
    use_flash_attn        = True,
    dropout               = 0.0,
    use_fp8               = False,
    hybrid_ratio          = 3,
    gdn_head_dim          = 64,
)

TASK_MAP = {
    "nq_open"        : ("nq_open",         1),
    "boolq"          : ("boolq",           0),
    "lambada_openai" : ("lambada_openai",  0),
    "piqa"           : ("piqa",            0),
    "mmlu"           : ("mmlu",            5),
    "arc_easy"       : ("arc_easy",        5),
    "arc_challenge"  : ("arc_challenge",  25),
    "hellaswag"      : ("hellaswag",      10),
    "winogrande"     : ("winogrande",      5),
    "triviaqa"       : ("triviaqa",        0),
    "openbookqa"     : ("openbookqa",      0),
    "sciq"           : ("sciq",            0),
    "copa"           : ("copa",            0),
    "race"           : ("race",            0),
    "commonsense_qa" : ("commonsense_qa",  0),
}

TASKS_ALL = list(TASK_MAP.keys())

RANDOM_BASELINES = {
    "piqa"           : 0.50,
    "triviaqa"       : 0.00,
    "mmlu"           : 0.25,
    "arc_easy"       : 0.25,
    "arc_challenge"  : 0.25,
    "hellaswag"      : 0.25,
    "winogrande"     : 0.50,
    "nq_open"        : 0.00,
    "boolq"          : 0.50,
    "lambada_openai" : 0.00,
    "openbookqa"     : 0.25,
    "sciq"           : 0.25,
    "copa"           : 0.50,
    "race"           : 0.25,
    "commonsense_qa" : 0.20,
}


class NaylisGDNLM(LM):

    def __init__(self, model, tokenizer, device, batch_size=4, max_seq_len=1024):
        super().__init__()
        self._model      = model
        self._tokenizer  = tokenizer
        self._device     = device
        self._batch_size = batch_size
        self._max_length = max_seq_len
        self._dtype      = torch.bfloat16 if device == "cuda" else torch.float32

    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    @property
    def accelerator(self):
        return None

    @property
    def tokenizer_name(self) -> str:
        return getattr(self._tokenizer, "name_or_path", TOKENIZER_ID)

    @property
    def chat_template(self) -> str:
        return ""

    def apply_chat_template(self, chat_history: list) -> str:
        return " ".join(m.get("content", "") for m in chat_history)

    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_token_id or 0

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def tok_decode(self, tokens) -> str:
        return self._tokenizer.decode(tokens)

    def _encode_pair(self, context: str, continuation: str):
        ctx_ids  = self.tok_encode(context) if context else []
        cont_ids = self.tok_encode(continuation)
        if not cont_ids:
            cont_ids = self.tok_encode(" " + continuation)
        full    = ctx_ids + cont_ids
        if len(full) > self._max_length:
            full    = full[-self._max_length:]
            ctx_len = max(1, len(full) - len(cont_ids))
        else:
            ctx_len = len(ctx_ids)
        return full, ctx_len, len(cont_ids)

    @torch.no_grad()
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        results = []
        pad_id  = self._tokenizer.pad_token_id or self.eot_token_id

        for i in tqdm(range(0, len(requests), self._batch_size),
                      desc="  loglikelihood", unit="batch",
                      dynamic_ncols=True, leave=False):
            batch_reqs = list(requests[i : i + self._batch_size])
            batch_data = [self._encode_pair(*req.args) for req in batch_reqs]

            max_len   = max(len(d[0]) for d in batch_data)
            input_ids = torch.full(
                (len(batch_data), max_len), pad_id,
                dtype=torch.long, device=self._device,
            )
            for j, (full_ids, _, _) in enumerate(batch_data):
                input_ids[j, :len(full_ids)] = torch.tensor(
                    full_ids, dtype=torch.long, device=self._device)

            with torch.amp.autocast(self._device, dtype=self._dtype,
                                    enabled=(self._device == "cuda")):
                logits, _, _ = self._model(input_ids)

            log_probs = F.log_softmax(logits, dim=-1)

            for j, (full_ids, ctx_len, cont_len) in enumerate(batch_data):
                cont_start  = ctx_len
                cont_end    = min(ctx_len + cont_len, max_len)
                target_ids  = torch.tensor(
                    full_ids[cont_start:cont_end],
                    dtype=torch.long, device=self._device,
                )
                logit_slice = log_probs[j, cont_start - 1 : cont_end - 1]

                if logit_slice.shape[0] == 0 or target_ids.shape[0] == 0:
                    results.append((float("-inf"), False))
                    continue

                n         = min(logit_slice.shape[0], target_ids.shape[0])
                token_ll  = logit_slice[:n].gather(
                    1, target_ids[:n].unsqueeze(1)
                ).squeeze(1)
                is_greedy = (logit_slice[:n].argmax(dim=-1) == target_ids[:n]).all().item()
                results.append((token_ll.sum().item(), bool(is_greedy)))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests) -> List[float]:
        results = []
        for req in requests:
            ids      = self.tok_encode(req.args[0])
            if not ids:
                results.append(0.0)
                continue
            total_ll = 0.0
            stride   = self._max_length

            for start in range(0, len(ids), stride):
                chunk = ids[max(0, start - 1) : start + stride]
                if len(chunk) < 2:
                    continue
                inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=self._device)
                tgt = torch.tensor(chunk[1:],   dtype=torch.long, device=self._device)
                with torch.amp.autocast(self._device, dtype=self._dtype,
                                        enabled=(self._device == "cuda")):
                    logits, _, _ = self._model(inp)
                score_from = 1 if start > 0 else 0
                lp         = F.log_softmax(logits[0], dim=-1)
                total_ll  += lp[score_from:].gather(
                    1, tgt[score_from:].unsqueeze(1)
                ).squeeze(1).sum().item()

            results.append(total_ll)
        return results

    @torch.no_grad()
    def generate_until(self, requests) -> List[str]:
        results = []
        for req in tqdm(requests, desc="  generate_until", unit="q", dynamic_ncols=True):
            ctx, gen_kwargs = req.args
            until    = gen_kwargs.get("until", [self._tokenizer.eos_token])
            max_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temp     = gen_kwargs.get("temperature", 0.0)
            top_p    = gen_kwargs.get("top_p", None)

            ids = self.tok_encode(ctx)
            ids = ids[-(self._max_length - max_toks):]
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)

            stop_ids = list({self.eot_token_id} | {
                self.tok_encode(s)[0]
                for s in until
                if s and len(self.tok_encode(s)) == 1
            })

            with torch.amp.autocast(self._device, dtype=self._dtype,
                                    enabled=(self._device == "cuda")):
                out_ids = self._model.generate(
                    input_ids,
                    max_new_tokens = max_toks,
                    temperature    = temp,
                    top_p          = top_p,
                    eos_token_id   = stop_ids[0] if len(stop_ids) == 1 else self.eot_token_id,
                )

            gen_text = self._tokenizer.decode(
                out_ids[0, input_ids.shape[1]:].tolist(),
                skip_special_tokens=True,
            )
            for stop in until:
                if stop and stop in gen_text:
                    gen_text = gen_text[:gen_text.index(stop)]

            results.append(gen_text.strip())
        return results


def load_tokenizer() -> AutoTokenizer:
    print(f"  Tokenizer : {TOKENIZER_ID}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_path: str, device: str) -> NaylisGDN:
    print(f"\n  Chargement : {model_path}")
    if not os.path.exists(model_path):
        sys.exit(f"ERREUR : fichier introuvable → {model_path}")

    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.endswith("_extra_state")}

    cfg_found = {}
    cfg_src   = "MODEL_CFG (défaut)"
    if "model_config" in ckpt:
        cfg_found = ckpt["model_config"]
        cfg_src   = "checkpoint .pt"
    else:
        info_path = model_path.replace(".pt", "_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                cfg_found = json.load(f).get("config", {})
            cfg_src = "_info.json"

    for k in _ARCH_KEYS:
        if k in cfg_found:
            MODEL_CFG[k] = cfg_found[k]

    emb_w = state.get("token_embeddings.weight")
    if emb_w is not None:
        MODEL_CFG["vocab_size"] = emb_w.shape[0]

    MODEL_CFG["use_fp8"] = False

    print(f"  Config source  : {cfg_src}")
    print(f"  embed={MODEL_CFG['embed_dim']}  layers={MODEL_CFG['num_layers']}  "
          f"heads={MODEL_CFG['num_heads']}  kv={MODEL_CFG['n_kv_heads']}")
    print(f"  vocab_size={MODEL_CFG['vocab_size']}")

    model = NaylisGDN(**MODEL_CFG)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Clés manquantes  : {len(missing)}")
    if unexpected:
        print(f"  Clés inattendues : {len(unexpected)}")

    model.to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Modèle chargé : {params:.1f}M params")
    return model


def main():
    parser = argparse.ArgumentParser(description="NaylisGDN — Benchmark Suite")
    parser.add_argument("--model",       default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tasks",       default="all")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--output",      default="./benchmark_results.json")
    parser.add_argument("--device",      default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device

    print("\n" + "=" * 65)
    print("  NaylisGDN — Benchmark Suite  [few-shot industrie]")
    print("=" * 65)
    print(f"  Device      : {device}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"  VRAM        : {vram:.1f} GB")
    print(f"  Modèle      : {args.model}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  max_seq_len : {MODEL_CFG['max_seq_len']}")

    if args.tasks.strip().lower() == "all":
        task_keys = TASKS_ALL
    else:
        task_keys = [t.strip().lower() for t in args.tasks.split(",")]

    tokenizer = load_tokenizer()
    MODEL_CFG.setdefault("vocab_size", len(tokenizer))
    if MODEL_CFG["vocab_size"] is None:
        MODEL_CFG["vocab_size"] = len(tokenizer)

    model = load_model(args.model, device)
    lm    = NaylisGDNLM(
        model, tokenizer, device,
        batch_size  = args.batch_size,
        max_seq_len = MODEL_CFG["max_seq_len"],
    )

    all_results = {}
    t0_total    = time.time()

    for key in task_keys:
        if key not in TASK_MAP:
            print(f"  Tâche inconnue : {key} — ignorée")
            continue

        task_name, default_fs = TASK_MAP[key]
        fs = args.num_fewshot if args.num_fewshot is not None else default_fs

        print(f"\n{'─' * 55}")
        print(f"  Tâche : {key}  ({fs}-shot)")
        t0 = time.time()

        try:
            results  = simple_evaluate(
                model       = lm,
                tasks       = [task_name],
                num_fewshot = fs,
                batch_size  = args.batch_size,
                log_samples = False,
            )
            task_res = results["results"].get(task_name, {})
            acc      = (
                task_res.get("acc,none") or
                task_res.get("acc_norm,none") or
                task_res.get("exact_match,none")
            )
            baseline = RANDOM_BASELINES.get(key)

            if acc is not None:
                baseline_str = f"   baseline : {baseline*100:.2f}%" if baseline is not None else ""
                print(f"  acc      : {acc*100:.2f}%{baseline_str}")
            else:
                print("  acc      : N/A")
            print(f"  Temps    : {time.time() - t0:.1f}s")

            all_results[key] = task_res

        except Exception as e:
            print(f"  ERREUR : {e}")
            all_results[key] = {"error": str(e)}

    elapsed = time.time() - t0_total
    print(f"\n{'='*55}")
    print(f"  DONE — {len(all_results)} tâches en {elapsed/60:.1f}min")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Résultats → {args.output}")


if __name__ == "__main__":
    main()
