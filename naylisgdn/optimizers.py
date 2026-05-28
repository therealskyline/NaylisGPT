import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 3,
        weight_decay: float = 0.0,
        use_mars: bool = True,
        mars_gamma: float = 0.025,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
            use_mars=use_mars, mars_gamma=mars_gamma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr         = group["lr"]
            momentum   = group["momentum"]
            nesterov   = group["nesterov"]
            ns_steps   = group["ns_steps"]
            wd         = group["weight_decay"]
            use_mars   = group.get("use_mars", True)
            mars_gamma = group.get("mars_gamma", 0.025)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    continue

                state = self.state[p]

                if use_mars:
                    if "prev_grad" not in state:
                        state["prev_grad"] = torch.zeros_like(g)
                    prev_g    = state["prev_grad"]
                    norm_g    = g.norm() + 1e-8
                    norm_prev = prev_g.norm() + 1e-8
                    c_t       = torch.clamp(
                        (mars_gamma / (1.0 - mars_gamma)) * (norm_g / norm_prev), max=1.0
                    )
                    g = g + c_t * (g - prev_g)
                    state["prev_grad"].copy_(p.grad)

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = (g + momentum * buf) if nesterov else buf

                g     = zeropower_via_newtonschulz5(g, steps=ns_steps)
                scale = max(g.size(0), g.size(1)) ** 0.5
                g     = g * scale

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple,
    eps: float,
    device: str = "cuda",
) -> tuple:
    MUON_EXCLUDE = {
        "token_embeddings.weight",
        "output_head.weight",
        "position_embeddings.weight",
    }

    muon_params, adamw_decay, adamw_nodecay = [], [], []

    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if pn in MUON_EXCLUDE:
            (adamw_decay if p.dim() >= 2 else adamw_nodecay).append(p)
            continue
        if p.dim() >= 2 and pn.startswith("blocks."):
            muon_params.append(p)
        elif p.dim() < 2 and pn.startswith("blocks."):
            adamw_nodecay.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    lr_muon  = lr * 5.0
    muon_opt = Muon(
        [{"params": muon_params, "is_muon": True}],
        lr=lr_muon, momentum=0.95, nesterov=True,
        ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025,
    )
    muon_opt.param_groups[0]["is_muon"] = True

    _is_cuda = device.startswith("cuda")
    adamw_opt = torch.optim.AdamW(
        [
            {"params": adamw_decay,   "weight_decay": weight_decay, "is_muon": False},
            {"params": adamw_nodecay, "weight_decay": 0.0,          "is_muon": False},
        ],
        lr=lr, betas=betas, eps=eps,
        fused=_is_cuda,
        capturable=_is_cuda,
    )

    print(f"\nOptimizer Muon+MARS + AdamW fused :")
    print(f"  Muon  : {len(muon_params)} tenseurs  lr={lr_muon:.2e}")
    print(f"  AdamW : {len(adamw_decay)} decay + {len(adamw_nodecay)} no-decay  lr={lr:.2e}")

    return muon_opt, adamw_opt
