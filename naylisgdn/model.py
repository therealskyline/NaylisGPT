import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from naylisgdn.norm import RMSNorm
from naylisgdn.attention import KVCache
from naylisgdn.transformer_block import TransformerBlock
from naylisgdn.gdn_block import GDNBlock, GDNState

_LIGER_LCE = None
try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    _LIGER_LCE = LigerFusedLinearCrossEntropyLoss()
    print("  ⚡ Liger-Kernel : Fused Linear Cross-Entropy (chunked, no logit tensor)")
except ImportError:
    pass

BlockState = Optional[Union[KVCache, GDNState]]


class NaylisGDN(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 1280,
        num_heads: int = 20,
        num_layers: int = 24,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: int = 10000,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 1024,
        use_swiglu: bool = True,
        n_kv_heads: Optional[int] = None,
        use_qk_norm: bool = False,
        soft_cap: Optional[float] = None,
        use_flash_attn: bool = True,
        use_fp8: bool = False,
        hybrid_ratio: int = 3,
        gdn_head_dim: Optional[int] = None,
        gdn_v_heads: Optional[int] = None,
        attn_head_dim: Optional[int] = None,
        use_moe: bool = False,
        num_experts: int = 16,
        top_k_experts: int = 2,
        shared_experts: int = 2,
        expert_hidden_dim: Optional[int] = None,
        moe_aux_coeff: float = 0.01,
        use_moh: bool = False,
        moh_shared_heads: Optional[int] = None,
        moh_top_k_routed: Optional[int] = None,
        mtp_steps: int = 0,
        mtp_weight: float = 0.3,
    ):
        super().__init__()

        assert vocab_size > 0 and embed_dim > 0 and num_layers > 0 and max_seq_len > 0
        assert hybrid_ratio >= 1

        if n_kv_heads is not None:
            assert num_heads % n_kv_heads == 0
        if use_rope and use_yarn:
            assert 0.1 <= yarn_scale <= 16.0
        if soft_cap is not None:
            assert 0 < soft_cap <= 100

        self.vocab_size            = vocab_size
        self.embed_dim             = embed_dim
        self.num_heads             = num_heads
        self.num_layers            = num_layers
        self.max_seq_len           = max_seq_len
        self.use_rope              = use_rope
        self.rope_base             = rope_base
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len
        self.use_swiglu            = use_swiglu
        self.n_kv_heads            = n_kv_heads
        self.use_qk_norm           = use_qk_norm
        self.soft_cap              = soft_cap
        self.use_flash_attn        = use_flash_attn
        self.use_fp8               = use_fp8
        self.hybrid_ratio          = hybrid_ratio
        self.gdn_head_dim          = gdn_head_dim if gdn_head_dim is not None else (embed_dim // num_heads)
        self.gdn_v_heads           = gdn_v_heads
        self.attn_head_dim         = attn_head_dim
        self.use_moe               = use_moe
        self.use_moh               = use_moh
        self.moe_aux_coeff         = moe_aux_coeff
        self.mtp_steps             = mtp_steps

        self.token_embeddings    = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = None if use_rope else nn.Embedding(max_seq_len, embed_dim)
        self.dropout             = nn.Dropout(dropout)

        self.blocks: nn.ModuleList     = nn.ModuleList()
        self.block_types: List[str]    = []

        for i in range(num_layers):
            is_gpt = (i % (hybrid_ratio + 1) == hybrid_ratio)

            if is_gpt:
                self.blocks.append(TransformerBlock(
                    embed_dim, num_heads, dropout,
                    use_rope=use_rope, max_seq_len=max_seq_len,
                    rope_base=rope_base,
                    use_yarn=use_yarn, yarn_scale=yarn_scale,
                    yarn_original_max_len=yarn_original_max_len,
                    use_swiglu=use_swiglu, n_kv_heads=n_kv_heads,
                    use_qk_norm=use_qk_norm, use_flash_attn=use_flash_attn,
                    soft_cap=soft_cap, use_fp8=use_fp8,
                    attn_head_dim=attn_head_dim,
                    use_moe=use_moe,
                    num_experts=num_experts, top_k_experts=top_k_experts,
                    shared_experts=shared_experts, expert_hidden_dim=expert_hidden_dim,
                    moe_aux_coeff=moe_aux_coeff,
                    use_moh=use_moh,
                    moh_shared_heads=moh_shared_heads,
                    moh_top_k_routed=moh_top_k_routed,
                ))
                self.block_types.append("gpt")
            else:
                self.blocks.append(GDNBlock(
                    embed_dim, num_heads, dropout,
                    use_swiglu=use_swiglu, use_fp8=use_fp8,
                    head_dim=self.gdn_head_dim,
                    v_heads=gdn_v_heads,
                ))
                self.block_types.append("gdn")

        self.num_gdn_blocks = self.block_types.count("gdn")
        self.num_gpt_blocks = self.block_types.count("gpt")

        self.ln_final    = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

        if mtp_steps > 0:
            from naylisgdn.mtp import MultiTokenPrediction
            self.mtp = MultiTokenPrediction(
                embed_dim, vocab_size,
                num_steps=mtp_steps,
                use_fp8=use_fp8,
                weight=mtp_weight,
            )
        else:
            self.mtp = None

        self.apply(self._init_weights)

        std_residual = 0.02 / math.sqrt(2 * num_layers)
        for name, module in self.named_modules():
            if name.endswith(".attention.out_proj") \
                    or name.endswith(".ffn.down_proj") \
                    or name.endswith(".o_proj"):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.normal_(module.weight, mean=0.0, std=std_residual)

        self.output_head.weight    = self.token_embeddings.weight
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        try:
            import transformer_engine.pytorch as te
            if isinstance(module, te.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return
        except ImportError:
            pass

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        past_states: Optional[List[BlockState]] = None,
        use_state_cache: bool = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[BlockState]]]:

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_embeds = self.token_embeddings(input_ids)
        if token_embeds.device.type == "cuda" and token_embeds.dtype == torch.float32:
            token_embeds = token_embeds.to(torch.bfloat16)

        if self.use_rope:
            x = self.dropout(token_embeds)
        else:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x         = self.dropout(token_embeds + self.position_embeddings(positions))

        gpt_mask = None
        has_manual_attn = any(
            (not getattr(b.attention, "use_flash_attn", True))
            for b, t in zip(self.blocks, self.block_types) if t == "gpt"
        )
        if has_manual_attn and seq_len > 1:
            gpt_mask = self._causal_mask[:seq_len, :seq_len]

        new_states: Optional[List[BlockState]] = [] if use_state_cache else None
        total_aux_loss = torch.zeros(1, device=device, dtype=token_embeds.dtype)
        use_gc = self.gradient_checkpointing and self.training and not use_state_cache

        for i, (block, btype) in enumerate(zip(self.blocks, self.block_types)):
            past = past_states[i] if past_states is not None else None

            if btype == "gpt":
                if use_gc:
                    from torch.utils.checkpoint import checkpoint as _ckpt

                    def _gpt_fwd(x_, _b=block, _p=past):
                        out, _, _aux = _b(
                            x_, mask=gpt_mask, past_kv=_p, use_kv_cache=False,
                            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                        )
                        return out

                    x       = _ckpt(_gpt_fwd, x, use_reentrant=False)
                    new_blk = None
                else:
                    x, new_blk, aux = block(
                        x, mask=gpt_mask, past_kv=past, use_kv_cache=use_state_cache,
                        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                    )
                    total_aux_loss = total_aux_loss + aux

            else:
                if use_gc:
                    from torch.utils.checkpoint import checkpoint as _ckpt

                    def _gdn_fwd(x_, _b=block, _p=past):
                        out, _ = _b(
                            x_, recurrent_state=None, use_recurrent=False,
                            cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q,
                        )
                        return out

                    x       = _ckpt(_gdn_fwd, x, use_reentrant=False)
                    new_blk = None
                else:
                    x, new_blk = block(
                        x, recurrent_state=past, use_recurrent=use_state_cache,
                        cu_seqlens=cu_seqlens_q, max_seqlen=max_seqlen_q,
                    )

            if use_state_cache:
                new_states.append(new_blk)

        hidden = x
        x      = self.ln_final(hidden)

        loss   = None
        logits = None

        if targets is not None:
            ignore_index = pad_token_id if pad_token_id is not None else -100

            if _LIGER_LCE is not None and self.soft_cap is None and self.training:
                loss = _LIGER_LCE(
                    x.view(-1, self.embed_dim),
                    self.output_head.weight,
                    targets.view(-1),
                    ignore_index=ignore_index,
                )
            else:
                logits = self.output_head(x)
                if self.soft_cap is not None:
                    logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    ignore_index=ignore_index,
                )

            if self.use_moe:
                loss = loss + self.moe_aux_coeff * total_aux_loss.squeeze()

            if self.mtp is not None and self.training:
                mtp_loss = self.mtp(
                    hidden, targets,
                    embed_fn=self.token_embeddings,
                    pad_token_id=pad_token_id,
                )
                loss = loss + mtp_loss

        else:
            logits = self.output_head(x)
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap

        return logits, loss, new_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()

        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]

        prefill_logits, _, past_states = self.forward(input_ids, use_state_cache=True)
        next_logits = prefill_logits[:, -1, :]

        for _ in range(max_new_tokens):
            logits = next_logits

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    k         = min(top_k, logits.size(-1))
                    topk_v, _ = torch.topk(logits, k)
                    logits    = logits.masked_fill(logits < topk_v[:, [-1]], float("-inf"))

                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                    sorted_probs     = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove_mask      = (cumulative_probs - sorted_probs) >= top_p
                    sorted_logits    = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            decode_logits, _, past_states = self.forward(
                next_token, past_states=past_states, use_state_cache=True
            )
            next_logits = decode_logits[:, -1, :]

        if was_training:
            self.train()
        return input_ids

    def resize_token_embeddings(self, new_vocab_size: int):
        if new_vocab_size == self.vocab_size:
            return
        old_emb = self.token_embeddings
        self.token_embeddings = nn.Embedding(new_vocab_size, self.embed_dim)
        n = min(old_emb.num_embeddings, new_vocab_size)
        with torch.no_grad():
            self.token_embeddings.weight.data[:n] = old_emb.weight.data[:n]
        self.output_head        = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight
        self.vocab_size         = new_vocab_size

    def count_parameters(self) -> dict:
        token_params = self.token_embeddings.weight.numel()
        pos_params   = self.position_embeddings.weight.numel() if self.position_embeddings else 0
        gdn_params   = sum(
            p.numel() for b, t in zip(self.blocks, self.block_types)
            if t == "gdn" for p in b.parameters()
        )
        gpt_params   = sum(
            p.numel() for b, t in zip(self.blocks, self.block_types)
            if t == "gpt" for p in b.parameters()
        )
        ln_params    = sum(p.numel() for p in self.ln_final.parameters())
        mtp_params   = sum(p.numel() for p in self.mtp.parameters()) if self.mtp else 0
        total        = token_params + pos_params + gdn_params + gpt_params + ln_params + mtp_params
        return {
            "token_embeddings": token_params,
            "pos_embeddings":   pos_params,
            "gdn_blocks":       gdn_params,
            "gpt_blocks":       gpt_params,
            "final_ln":         ln_params,
            "mtp":              mtp_params,
            "output_head":      0,
            "total":            total,
            "num_gdn":          self.num_gdn_blocks,
            "num_gpt":          self.num_gpt_blocks,
        }

    def get_config(self) -> dict:
        return {
            "vocab_size":            self.vocab_size,
            "embed_dim":             self.embed_dim,
            "num_heads":             self.num_heads,
            "num_layers":            self.num_layers,
            "max_seq_len":           self.max_seq_len,
            "use_rope":              self.use_rope,
            "rope_base":             self.rope_base,
            "use_yarn":              self.use_yarn,
            "yarn_scale":            self.yarn_scale,
            "yarn_original_max_len": self.yarn_original_max_len,
            "use_swiglu":            self.use_swiglu,
            "n_kv_heads":            self.n_kv_heads,
            "use_qk_norm":           self.use_qk_norm,
            "soft_cap":              self.soft_cap,
            "use_flash_attn":        self.use_flash_attn,
            "use_fp8":               self.use_fp8,
            "hybrid_ratio":          self.hybrid_ratio,
            "gdn_head_dim":          self.gdn_head_dim,
            "use_moe":               self.use_moe,
            "use_moh":               self.use_moh,
            "mtp_steps":             self.mtp_steps,
        }

    def print_layout(self):
        moe_tag = "+MoE" if self.use_moe else ""
        moh_tag = "+MoH" if self.use_moh else ""
        mtp_tag = f"+MTP({self.mtp_steps})" if self.mtp_steps > 0 else ""
        print(f"\nLayout NaylisGDN — {self.num_layers} couches ({self.hybrid_ratio}:1 GDN/GPT){moe_tag}{moh_tag}{mtp_tag}")
        print(f"  GDN blocks : {self.num_gdn_blocks}  ({self.num_gdn_blocks/self.num_layers*100:.0f}%)")
        print(f"  GPT blocks : {self.num_gpt_blocks}  ({self.num_gpt_blocks/self.num_layers*100:.0f}%){moe_tag}{moh_tag}")
        row = ""
        for i, t in enumerate(self.block_types):
            row += f"[{i:02d}:{'G' if t=='gdn' else 'A'}] "
            if (i + 1) % 8 == 0:
                print(f"  {row}")
                row = ""
        if row:
            print(f"  {row}")
        print(f"  G=GDN  A=Attention/GPT")
