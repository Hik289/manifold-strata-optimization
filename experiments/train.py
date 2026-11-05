"""STRATA RUNNING-wave1 主训练脚本 (CIFAR-100 / CompactViT)。

支持: adamw / galore / single_grassmann / independent_grassmann / strata
§5.3 全量日志: 每 epoch train/val/test loss+acc, lr, grad norm, level energy,
peak mem, wall-clock; 每 flag update: δ_nest, principal angle, level energy, 内外层漂移。

用法:
    python train.py --method strata --epochs 1000 --R 64 --out runs/strata
"""

from __future__ import annotations
from pathlib import Path

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
from model import CompactViT, count_params  # noqa: E402
from data import get_cifar100_loaders  # noqa: E402
from state_budget import budget_report  # noqa: E402
from strata_pkg import diagnostics as diag  # noqa: E402
from strata_pkg.strata_adamw import STRATAAdamW  # noqa: E402
from strata_pkg.baselines import (  # noqa: E402
    SingleGrassmannAdamW, GaLoreAdamW, IndependentGrassmannAdamW,
)

JST = timezone(timedelta(hours=9))


def build_optimizer(method, model, args):
    """构造 optimizer。所有低秩方法 matched max rank R + 相同 lr/wd。"""
    R = args.R
    ranks = [R // 4, R // 2, R]         # EDA 建议 (R/4, R/2, R)
    rhos = [0.999, 0.98, 0.9]           # EDA 建议
    K = len(ranks)
    common = dict(lr=args.lr, weight_decay=args.wd, seed=args.seed)
    params = model.parameters()
    if method == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd,
                                 betas=(0.9, 0.999)), {"type": "full"}
    if method == "galore":
        return GaLoreAdamW(params, lr=args.lr, R=R, tau=args.galore_tau, scale=args.galore_scale,
                           weight_decay=args.wd), {"R": R, "tau": args.galore_tau}
    if method == "single_grassmann":
        return SingleGrassmannAdamW(params, R=R, rho=0.999, beta1=0.9, beta2=0.999,
                                    a=1.0, a_perp=1.0, xi=args.xi, tau=args.tau,
                                    gamma=1e-6, p=args.p, **common), {"R": R}
    if method == "independent_grassmann":
        return IndependentGrassmannAdamW(params, ranks=ranks, rhos=rhos, a_perp=1.0,
                                         xi=args.xi, tau=args.tau, gamma=1e-6, p=args.p,
                                         **common), {"ranks": ranks}
    if method == "strata":
        # 内层漂移极小 (EDA) => 内层可更长 τ / 更小 ξ。这里用统一 τ/ξ 起步,
        # 层间学习率差 a_ks 设非平凡值以保证 Γᵃ>0 (Corollary 18.3: 否则旗几何不参与更新)
        a_ks = list(args.a_ks) if args.a_ks else [1.0, 1.0, 1.0]
        return STRATAAdamW(params, lr=args.lr, ranks=ranks, rhos=rhos,
                           beta1s=[0.9, 0.9, 0.9], beta2s=[0.999, 0.999, 0.999],
                           a_ks=a_ks, a_perp=args.a_perp, xi=args.xi, tau=args.tau,
                           gamma=1e-6, p=args.p, weight_decay=args.wd,
                           seed=args.seed), {"ranks": ranks, "rhos": rhos, "a_ks": a_ks}
    raise ValueError(method)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


_NULL_CACHE = {}  # (n, *ranks) -> 随机零假设 δ_nest (只依赖维度, 全程复用)


def collect_strata_diagnostics(optimizer, ranks, prev_state):
    """从 STRATA/Grassmann optimizer 抽取 §5.3 hook 7-10 全量诊断。

    prev_state: dict[pid -> {"cum": [累计基...], "inc": [增量块...]}], 用于算漂移。
    返回 (stats, new_state)。

    加固 (RUNNING-wave1 授权):
    - hook 8 显式 principal-angle 漂移: 每层【增量块】U^[k](t) vs U^[k](t-1) 的 max 主角
      (度)。这是 H2 内外层 10-45× 分离 + §4.8 δ_t/e_track 的直接数据源。用增量块而非
      累计基, 因累计基内层⊂外层不独立。
    - level_energy None 兜底: 能量塌到~0 如实记为实测极小数值 + collapse flag,
      不丢 None (对应失败判据 §11.2-3 中间层能量≈0, 有据可查)。
    - Frobenius 投影漂移 ‖P_k(t)-P_k(t-1)‖_F 保留 (累计基, 与旧口径兼容)。
    """
    K = len(ranks)
    nesting_gaps, orthos = [], []
    level_energies = [[] for _ in range(K)]
    frob_drifts = [[] for _ in range(K)]       # 累计投影 Frobenius 漂移
    angle_drifts = [[] for _ in range(K)]      # 增量块 principal-angle 漂移 (度)
    energy_collapse = [0 for _ in range(K)]    # 各层能量塌陷计数
    indep_dnest = []                            # implementation#1: 独立最优子空间 δ_nest
    indep_dnest_norm = []                       # δ_nest / 随机零假设 (归一, <1 = 有嵌套结构)
    ENERGY_FLOOR = 1e-8
    new_state = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if "U" not in st:
                continue
            U = st["U"]
            if not isinstance(U, torch.Tensor):
                continue
            pid = id(p)
            slices = st.get("slices", None)
            cum = [U[:, :ranks[k]].detach().clone() for k in range(K)]
            if slices is not None:
                inc = [U[:, slices[k]].detach().clone() for k in range(K)]
            else:
                # baseline (Single-Grassmann K=1 / Independent) 无 slices: 用累计块近似
                inc = cum
            nesting_gaps.append(diag.nesting_gap(cum))
            R = U.shape[1]
            orthos.append(float((U.t() @ U - torch.eye(R, dtype=U.dtype, device=U.device)).norm()))

            # implementation#1: 大尺度 δ_nest —— 从多尺度 sketch 恢复各尺度独立最优子空间
            # V̂_k=Top_{r_k}(C̃_k), 算它们之间的嵌套缺口 (不是 STRATA 内部旗恒 0 的那个)。
            # 仅 STRATA 有多尺度 Y/Omega (K 个); baseline 跳过。
            Ys = st.get("Y", None)
            Oms = st.get("Omega", None)
            if isinstance(Ys, list) and len(Ys) == K and isinstance(Oms, list):
                try:
                    gamma = group.get("gamma", 1e-6)
                    Cs = [diag._ctilde_from_sketch(Ys[k], Oms[k], gamma) for k in range(K)]
                    dn, _, _ = diag.independent_nesting_gap(Cs, ranks)
                    indep_dnest.append(dn)
                    # 随机零假设只依赖 (ranks, n) 与实际数据无关 → 按 n 缓存, 避免每矩阵重算
                    n_dim = U.shape[0]
                    ck = (n_dim,) + tuple(ranks)
                    if ck not in _NULL_CACHE:
                        _NULL_CACHE[ck] = diag.random_nesting_null(
                            ranks, n_dim, U.dtype, U.device, n_samples=8, seed=1234)
                    null = _NULL_CACHE[ck]
                    if null > 1e-12:
                        indep_dnest_norm.append(dn / null)
                except Exception:
                    pass

            # level energy (None 兜底为实测极小值 + collapse flag)
            le = st.get("last_stats", {}).get("level_energy", None)
            for k in range(K):
                if le is not None and k < len(le) and le[k] is not None:
                    val = float(le[k])
                else:
                    val = 0.0
                level_energies[k].append(val)
                if val < ENERGY_FLOOR:
                    energy_collapse[k] += 1

            # 漂移 (需上一次 state)
            if pid in prev_state:
                pcum = prev_state[pid]["cum"]
                pinc = prev_state[pid]["inc"]
                for k in range(K):
                    Pk_now = cum[k] @ cum[k].t()
                    Pk_prev = pcum[k] @ pcum[k].t()
                    frob_drifts[k].append(float((Pk_now - Pk_prev).norm()))
                    # principal angle (增量块, 度): max 主角
                    try:
                        ang = diag.principal_angles(inc[k], pinc[k])
                        angle_drifts[k].append(float(ang.max()) * 180.0 / math.pi)
                    except Exception:
                        pass
            new_state[pid] = {"cum": cum, "inc": inc}

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else None
    stats = {
        "nesting_gap_mean": _mean(nesting_gaps),
        "orthogonality_mean": _mean(orthos),
        "indep_dnest_mean": _mean(indep_dnest),            # implementation#1: 独立最优 δ_nest
        "indep_dnest_norm_mean": _mean(indep_dnest_norm),  # /随机零假设 (<1=有嵌套结构)
        "level_energy_mean": [_mean(level_energies[k]) for k in range(K)],
        "level_energy_collapse_count": energy_collapse,
        "layer_frob_drift_mean": [_mean(frob_drifts[k]) for k in range(K)],
        "layer_angle_drift_deg_mean": [_mean(angle_drifts[k]) for k in range(K)],
    }
    return stats, new_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["adamw", "galore", "single_grassmann",
                             "independent_grassmann", "strata"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--xi", type=float, default=1e-3)
    ap.add_argument("--tau", type=int, default=50)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--a_ks", type=float, nargs="*", default=None)
    ap.add_argument("--a_perp", type=float, default=0.5)
    ap.add_argument("--galore_tau", type=int, default=200)
    ap.add_argument("--galore_scale", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flag_log_every", type=int, default=10, help="每多少 epoch 存一次 flag 诊断")
    ap.add_argument("--max_minutes", type=float, default=0, help=">0 时用于 timing 试跑早停")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        args.data_dir, batch_size=args.batch_size, seed=args.seed)
    model = CompactViT(num_classes=100).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer, opt_meta = build_optimizer(args.method, model, args)

    ranks = [args.R // 4, args.R // 2, args.R]
    K = len(ranks)
    budget = budget_report(model, args.R, ranks, K, args.p)

    # cosine schedule with warmup
    def lr_at(epoch):
        if epoch < args.warmup:
            return args.lr * (epoch + 1) / args.warmup
        prog = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))

    log_path = os.path.join(args.out, "log.jsonl")
    meta = {
        "project_id": "manifoldstrata", "phase": "RUNNING-wave1",
        "method": args.method, "task": "CIFAR-100 / CompactViT",
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
    t_start = time.time()
    best_val = 0.0

    with open(log_path, "w") as logf:
        for epoch in range(args.epochs):
            lr = lr_at(epoch)
            for g in optimizer.param_groups:
                g["lr"] = lr
            model.train()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ep_t0 = time.time()
            train_loss_sum, train_n, gnorm_sum, gnorm_cnt = 0.0, 0, 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)  # 只测不裁
                optimizer.step()
                train_loss_sum += loss.item() * x.size(0)
                train_n += x.size(0)
                gnorm_sum += float(gn); gnorm_cnt += 1
            train_loss = train_loss_sum / train_n
            ep_time = time.time() - ep_t0
            peak_mem = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0.0

            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            test_loss, test_acc = evaluate(model, test_loader, device, criterion)
            best_val = max(best_val, val_acc)

            rec = {
                "epoch": epoch, "lr": lr, "train_loss": train_loss,
                "val_loss": val_loss, "val_acc": val_acc,
                "test_loss": test_loss, "test_acc": test_acc,
                "grad_norm": gnorm_sum / max(1, gnorm_cnt),
                "epoch_time_s": ep_time, "peak_mem_gb": peak_mem,
            }
            # flag 诊断 (每 flag_log_every epoch)
            if is_flaglike and (epoch % args.flag_log_every == 0 or epoch == args.epochs - 1):
                fstats, prev_state = collect_strata_diagnostics(optimizer, ranks, prev_state)
                rec["flag_diag"] = fstats
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()

            if epoch % 50 == 0 or epoch == args.epochs - 1:
                print(f"[{args.method}] ep{epoch} train={train_loss:.4f} "
                      f"val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
                      f"peak={peak_mem:.2f}GB t={ep_time:.1f}s", flush=True)

            if args.max_minutes > 0 and (time.time() - t_start) / 60 > args.max_minutes:
                print(f"[TIMING] 到达 {args.max_minutes} 分钟, 早停于 epoch {epoch}", flush=True)
                meta["timing_stop_epoch"] = epoch
                break

    total_time = time.time() - t_start
    summary = {
        "method": args.method, "epochs_run": epoch + 1,
        "final_val_acc": val_acc, "final_test_acc": test_acc,
        "best_val_acc": best_val, "total_time_s": total_time,
        "avg_epoch_time_s": total_time / (epoch + 1),
        "peak_mem_gb": peak_mem, "state_budget": budget,
        "timestamp_end": datetime.now(JST).isoformat(),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {args.method}: final_test_acc={test_acc:.4f} "
          f"avg_epoch={total_time/(epoch+1):.1f}s total={total_time/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
