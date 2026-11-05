"""STRATA wave2 SciML 任务: Burgers PINN, 2000 epochs, 纯 PyTorch。

Burgers: u_t + u·u_x = ν·u_xx,  ν=0.01/π,  x∈[-1,1], t∈[0,1]
IC: u(0,x) = -sin(πx);  BC: u(t,±1)=0
PINN MLP: [2] -> 4×[hidden] -> [1], tanh。hidden 层是 2D 矩阵 → STRATA 旗分支。
metric: 相对 L2 (对参考解); 每 epoch 全量日志 + 全轨迹 δ_nest。

参考解: 用细网格有限差分预生成 (scipy 不需要, 自实现 Cole-Hopf 或 FD)。
这里用 Cole-Hopf 解析近似作为参考 (Burgers ν 小时的标准做法)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_budget import budget_report  # noqa: E402
from train import build_optimizer, collect_strata_diagnostics  # noqa: E402

JST = timezone(timedelta(hours=9))
NU = 0.01 / math.pi


class PINN(nn.Module):
    def __init__(self, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def cole_hopf_reference(nx=256, nt=100):
    """Cole-Hopf 变换求 Burgers 参考解 (ν=0.01/π, IC=-sin(πx))。
    返回 (X, T, U_ref) 网格。用数值积分算 Cole-Hopf。
    """
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    # Cole-Hopf: u = -2ν φ_x/φ, φ = ∫ exp(-(x-y)^2/(4νt) - cos(πy)/(2πν)) dy
    # 小 ν 下积分尖锐, 用高斯-埃尔米特近似较难; 改用细网格 FD 时间推进作为参考。
    return _fd_reference(nx, nt)


def _fd_reference(nx=512, nt_out=100):
    """有限差分参考解 (细网格显式推进, upwind 平流 + 中心扩散, 输出 nt_out 时间切片)。"""
    xf = np.linspace(-1, 1, nx)
    dx = xf[1] - xf[0]
    # CFL: 扩散 dt<dx²/2ν, 平流 dt<dx/max|u|. 取保守系数。
    dt = min(0.2 * dx * dx / NU, 0.4 * dx)
    steps = int(1.0 / dt) + 1
    dt = 1.0 / steps
    u = -np.sin(np.pi * xf)
    u[0] = 0.0; u[-1] = 0.0
    out_times = np.linspace(0, 1.0, nt_out)
    snaps = [u.copy()]
    tgrid = [0.0]
    next_save = 1
    for s in range(1, steps + 1):
        uxx = np.zeros_like(u)
        uxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx * dx)
        # upwind 平流: f=u²/2, 通量分裂按 u 符号
        ux_up = np.zeros_like(u)
        pos = u[1:-1] > 0
        back = (u[1:-1] - u[:-2]) / dx
        fwd = (u[2:] - u[1:-1]) / dx
        ux_up[1:-1] = np.where(pos, back, fwd)
        u = u + dt * (-u * ux_up + NU * uxx)
        u[0] = 0.0; u[-1] = 0.0
        tcur = s * dt
        while next_save < nt_out and tcur >= out_times[next_save] - 1e-12:
            snaps.append(u.copy())
            tgrid.append(out_times[next_save])
            next_save += 1
    while len(snaps) < nt_out:
        snaps.append(u.copy()); tgrid.append(1.0)
    U = np.stack(snaps, axis=0)  # (nt_out, nx)
    return xf, np.array(tgrid), U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["adamw", "galore", "single_grassmann",
                             "independent_grassmann", "strata"])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n_col", type=int, default=10000, help="配点数")
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--xi", type=float, default=1e-3)
    ap.add_argument("--tau", type=int, default=50)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--a_ks", type=float, nargs="*", default=None)
    ap.add_argument("--a_perp", type=float, default=0.2)
    ap.add_argument("--galore_tau", type=int, default=200)
    ap.add_argument("--galore_scale", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flag_log_every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    # 参考解
    xf, tgrid, Uref = _fd_reference(nx=256, nt_out=100)
    Xr, Tr = np.meshgrid(xf, tgrid)
    ref_x = torch.tensor(Xr.reshape(-1, 1), dtype=torch.float32, device=device)
    ref_t = torch.tensor(Tr.reshape(-1, 1), dtype=torch.float32, device=device)
    ref_u = torch.tensor(Uref.reshape(-1, 1), dtype=torch.float32, device=device)

    model = PINN(args.hidden, args.depth).to(device)
    n_params = count_params(model)
    optimizer, opt_meta = build_optimizer(args.method, model, args)
    ranks = [args.R // 4, args.R // 2, args.R]
    K = len(ranks)
    budget = budget_report(model, args.R, ranks, K, args.p)

    def lr_at(epoch):
        if epoch < args.warmup:
            return args.lr * (epoch + 1) / args.warmup
        prog = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))

    # 固定配点 (seed 决定)
    g = torch.Generator().manual_seed(args.seed)
    x_col = (torch.rand(args.n_col, 1, generator=g) * 2 - 1).to(device).requires_grad_(True)
    t_col = torch.rand(args.n_col, 1, generator=g).to(device).requires_grad_(True)
    # IC 点
    x_ic = (torch.rand(500, 1, generator=g) * 2 - 1).to(device)
    t_ic = torch.zeros(500, 1, device=device)
    u_ic = -torch.sin(math.pi * x_ic)
    # BC 点
    t_bc = torch.rand(500, 1, generator=g).to(device)
    x_bc = torch.cat([torch.full((250, 1), -1.0), torch.full((250, 1), 1.0)]).to(device)

    def pde_residual(x, t):
        u = model(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return u_t + u * u_x - NU * u_xx

    @torch.no_grad()
    def rel_l2():
        model.eval()
        pred = model(ref_x, ref_t)
        err = torch.linalg.norm(pred - ref_u) / (torch.linalg.norm(ref_u) + 1e-12)
        model.train()
        return float(err)

    meta = {
        "project_id": "manifoldstrata", "phase": "RUNNING-wave2",
        "method": args.method, "task": "Burgers PINN",
        "n_params": n_params, "opt_meta": opt_meta,
        "config": vars(args), "state_budget": budget,
        "timestamp_start": datetime.now(JST).isoformat(),
        "torch": torch.__version__,
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
            r = pde_residual(x_col, t_col)
            loss_pde = (r ** 2).mean()
            loss_ic = ((model(x_ic, t_ic) - u_ic) ** 2).mean()
            loss_bc = (model(x_bc, t_bc) ** 2).mean()
            loss = loss_pde + loss_ic + loss_bc
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            optimizer.step()
            ep_time = time.time() - ep_t0
            peak_mem = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0.0

            rl2 = rel_l2()
            best_rel = min(best_rel, rl2)
            rec = {
                "epoch": epoch, "lr": lr, "train_loss": float(loss.item()),
                "loss_pde": float(loss_pde.item()), "loss_ic": float(loss_ic.item()),
                "loss_bc": float(loss_bc.item()), "rel_l2": rl2,
                "grad_norm": float(gn), "epoch_time_s": ep_time, "peak_mem_gb": peak_mem,
            }
            if is_flaglike and (epoch % args.flag_log_every == 0 or epoch == args.epochs - 1):
                fstats, prev_state = collect_strata_diagnostics(optimizer, ranks, prev_state)
                rec["flag_diag"] = fstats
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()
            if epoch % 200 == 0 or epoch == args.epochs - 1:
                print(f"[{args.method}] ep{epoch} loss={loss.item():.4e} "
                      f"rel_l2={rl2:.4e} peak={peak_mem:.2f}GB t={ep_time*1000:.0f}ms", flush=True)

    total_time = time.time() - t_start
    summary = {
        "method": args.method, "epochs_run": args.epochs,
        "final_rel_l2": rl2, "best_rel_l2": best_rel,
        "total_time_s": total_time, "avg_epoch_time_s": total_time / args.epochs,
        "peak_mem_gb": peak_mem, "state_budget": budget,
        "timestamp_end": datetime.now(JST).isoformat(),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {args.method}: final_rel_l2={rl2:.4e} best={best_rel:.4e} "
          f"avg_epoch={total_time/args.epochs*1000:.0f}ms total={total_time/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
