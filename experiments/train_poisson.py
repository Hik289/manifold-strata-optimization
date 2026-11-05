"""G2 第二任务: 多频 1D Poisson PINN (与 Burgers 不同方程, 天然多尺度)。

-u''(x) = f(x),  x∈[0,1],  u(0)=u(1)=0
解析解: u(x) = sin(π x) + 0.1 sin(8π x) + 0.02 sin(24π x)   (低+中+高频三尺度)
=> f(x) = π²sin(πx) + 0.1·(8π)²sin(8πx) + 0.02·(24π)²sin(24πx)

三个频率成分 = 天然多时间尺度, 是 STRATA 嵌套旗设计动机的第二个独立验证。
复用 train.py optimizer builder + collect_strata_diagnostics。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_budget import budget_report  # noqa: E402
from train import build_optimizer, collect_strata_diagnostics  # noqa: E402

JST = timezone(timedelta(hours=9))
FREQS = [(1.0, 1.0), (8.0, 0.1), (24.0, 0.02)]  # (k, amp): 低/中/高频


def u_exact(x):
    u = torch.zeros_like(x)
    for k, a in FREQS:
        u = u + a * torch.sin(k * math.pi * x)
    return u


def f_source(x):
    f = torch.zeros_like(x)
    for k, a in FREQS:
        f = f + a * (k * math.pi) ** 2 * torch.sin(k * math.pi * x)
    return f


class MLP(nn.Module):
    def __init__(self, hidden=128, depth=5):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["adamw", "galore", "single_grassmann",
                             "independent_grassmann", "strata"])
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n_col", type=int, default=5000)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--xi", type=float, default=1e-3)
    ap.add_argument("--tau", type=int, default=50)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--a_ks", type=float, nargs="*", default=None)
    ap.add_argument("--a_perp", type=float, default=0.2)
    ap.add_argument("--galore_tau", type=int, default=200)
    ap.add_argument("--galore_scale", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flag_log_every", type=int, default=30)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    model = MLP(args.hidden, args.depth).to(device)
    optimizer, opt_meta = build_optimizer(args.method, model, args)
    ranks = [args.R // 4, args.R // 2, args.R]
    K = len(ranks)
    budget = budget_report(model, args.R, ranks, K, args.p)

    xref = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    uref = u_exact(xref)

    def lr_at(epoch):
        if epoch < args.warmup:
            return args.lr * (epoch + 1) / args.warmup
        prog = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))

    g = torch.Generator().manual_seed(args.seed)
    x_col = torch.rand(args.n_col, 1, generator=g).to(device).requires_grad_(True)
    x_bc = torch.tensor([[0.0], [1.0]], device=device)

    def residual(x):
        u = model(x)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return -u_xx - f_source(x)

    @torch.no_grad()
    def rel_l2():
        model.eval()
        pred = model(xref)
        err = torch.linalg.norm(pred - uref) / (torch.linalg.norm(uref) + 1e-12)
        model.train()
        return float(err)

    meta = {
        "project_id": "manifoldstrata", "phase": "RUNNING-wave2-G2",
        "method": args.method, "task": "Multi-freq Poisson PINN",
        "n_params": count_params(model), "opt_meta": opt_meta,
        "config": vars(args), "state_budget": budget,
        "timestamp_start": datetime.now(JST).isoformat(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    is_flaglike = args.method in ("strata", "single_grassmann", "independent_grassmann")
    prev_state = {}
    log_path = os.path.join(args.out, "log.jsonl")
    t_start = time.time()
    best_rel = 1e9
    with open(log_path, "w") as logf:
        for epoch in range(args.epochs):
            lr = lr_at(epoch)
            for gg in optimizer.param_groups:
                gg["lr"] = lr
            model.train()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ep_t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            r = residual(x_col)
            loss_pde = (r ** 2).mean()
            loss_bc = (model(x_bc) ** 2).mean()
            loss = loss_pde + loss_bc
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            optimizer.step()
            ep_time = time.time() - ep_t0
            peak_mem = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0.0
            rl2 = rel_l2()
            best_rel = min(best_rel, rl2)
            rec = {"epoch": epoch, "lr": lr, "train_loss": float(loss.item()),
                   "loss_pde": float(loss_pde.item()), "loss_bc": float(loss_bc.item()),
                   "rel_l2": rl2, "grad_norm": float(gn),
                   "epoch_time_s": ep_time, "peak_mem_gb": peak_mem}
            if is_flaglike and (epoch % args.flag_log_every == 0 or epoch == args.epochs - 1):
                fstats, prev_state = collect_strata_diagnostics(optimizer, ranks, prev_state)
                rec["flag_diag"] = fstats
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()
            if epoch % 300 == 0 or epoch == args.epochs - 1:
                print(f"[{args.method}] ep{epoch} loss={loss.item():.4e} rel_l2={rl2:.4e} "
                      f"t={ep_time*1000:.0f}ms", flush=True)
    total_time = time.time() - t_start
    summary = {"method": args.method, "epochs_run": args.epochs,
               "final_rel_l2": rl2, "best_rel_l2": best_rel,
               "total_time_s": total_time, "avg_epoch_time_s": total_time / args.epochs,
               "peak_mem_gb": peak_mem, "state_budget": budget,
               "timestamp_end": datetime.now(JST).isoformat()}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {args.method}: final_rel_l2={rl2:.4e} best={best_rel:.4e}", flush=True)


if __name__ == "__main__":
    main()
