"""CNN(ResNet-18)/MLP benchmark: CIFAR-10/100 + MNIST/Fashion-MNIST。

conv-2D 化在 optimizer 内 (见 strata_pkg)。复用 train.py 的 build_optimizer +
collect_strata_diagnostics (全轨迹 δ_nest)。judge: 末端 100ep 均值 test_acc。
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
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_cnn_mlp import ResNet18, MLP3, count_params  # noqa: E402
from state_budget import budget_report  # noqa: E402
from train import build_optimizer, collect_strata_diagnostics  # noqa: E402

JST = timezone(timedelta(hours=9))

DATASETS = {
    "cifar10": dict(cls=torchvision.datasets.CIFAR10, nc=10, ch=3, kind="cnn",
                    mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    "cifar100": dict(cls=torchvision.datasets.CIFAR100, nc=100, ch=3, kind="cnn",
                     mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    "mnist": dict(cls=torchvision.datasets.MNIST, nc=10, ch=1, kind="mlp",
                  mean=(0.1307,), std=(0.3081,)),
    "fashion": dict(cls=torchvision.datasets.FashionMNIST, nc=10, ch=1, kind="mlp",
                    mean=(0.2860,), std=(0.3530,)),
}


def get_loaders(dsname, data_dir, batch_size, seed, val_split=5000):
    d = DATASETS[dsname]
    if d["kind"] == "cnn":
        train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize(d["mean"], d["std"])])
    else:
        train_tf = T.Compose([T.ToTensor(), T.Normalize(d["mean"], d["std"])])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(d["mean"], d["std"])])
    Cls = d["cls"]
    full = Cls(data_dir, train=True, download=True, transform=train_tf)
    full_eval = Cls(data_dir, train=True, download=True, transform=eval_tf)
    test = Cls(data_dir, train=False, download=True, transform=eval_tf)
    n = len(full)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    vidx, tidx = perm[:val_split].tolist(), perm[val_split:].tolist()
    train_sub = torch.utils.data.Subset(full, tidx)
    val_sub = torch.utils.data.Subset(full_eval, vidx)
    tl = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                                     num_workers=4, pin_memory=True, drop_last=True,
                                     generator=torch.Generator().manual_seed(seed))
    vl = torch.utils.data.DataLoader(val_sub, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    tel = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    return tl, vl, tel, d


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    tot, cor, ls = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        ls += criterion(out, y).item() * x.size(0)
        cor += (out.argmax(1) == y).sum().item()
        tot += x.size(0)
    return ls / tot, cor / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["adamw", "galore", "single_grassmann", "independent_grassmann", "strata"])
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--xi", type=float, default=1e-3)
    ap.add_argument("--tau", type=int, default=50)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--a_ks", type=float, nargs="*", default=None)
    ap.add_argument("--a_perp", type=float, default=0.2)
    ap.add_argument("--galore_tau", type=int, default=200)
    ap.add_argument("--galore_scale", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=str(Path(__file__).resolve().parents[1] / 'data'))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flag_log_every", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    tl, vl, tel, dmeta = get_loaders(args.dataset, args.data_dir, args.batch_size, args.seed)
    if dmeta["kind"] == "cnn":
        model = ResNet18(num_classes=dmeta["nc"], in_ch=dmeta["ch"]).to(device)
    else:
        model = MLP3(in_dim=28 * 28 * dmeta["ch"], hidden=args.hidden, num_classes=dmeta["nc"]).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer, opt_meta = build_optimizer(args.method, model, args)
    ranks = [args.R // 4, args.R // 2, args.R]
    K = len(ranks)
    budget = budget_report(model, args.R, ranks, K, args.p)

    def lr_at(ep):
        if ep < args.warmup:
            return args.lr * (ep + 1) / args.warmup
        prog = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))

    meta = {
        "project_id": "manifoldstrata", "phase": "RUNNING-wave3-cnnmlp",
        "method": args.method, "task": f"{args.dataset} / {dmeta['kind'].upper()}",
        "conv2d_note": "4D conv reshape 到 2D [out,in*kh*kw] 走旗分支 (implementation 约定, 全方法一致)",
        "n_params": n_params, "opt_meta": opt_meta, "config": vars(args),
        "state_budget": budget, "timestamp_start": datetime.now(JST).isoformat(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    is_flaglike = args.method in ("strata", "single_grassmann", "independent_grassmann")
    prev_state = {}
    log_path = os.path.join(args.out, "log.jsonl")
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
            tls, tn = 0.0, 0
            gnsum, gncnt = 0.0, 0
            for x, y in tl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)  # 只测不裁
                optimizer.step()
                tls += loss.item() * x.size(0); tn += x.size(0)
                gnsum += float(gn); gncnt += 1
            train_loss = tls / tn
            grad_norm = gnsum / max(1, gncnt)
            ep_time = time.time() - ep_t0
            peak_mem = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0.0
            val_loss, val_acc = evaluate(model, vl, device, criterion)
            test_loss, test_acc = evaluate(model, tel, device, criterion)
            best_val = max(best_val, val_acc)
            rec = {"epoch": epoch, "lr": lr, "train_loss": train_loss,
                   "val_loss": val_loss, "val_acc": val_acc,
                   "test_loss": test_loss, "test_acc": test_acc,
                   "grad_norm": grad_norm,
                   "epoch_time_s": ep_time, "peak_mem_gb": peak_mem}
            if is_flaglike and (epoch % args.flag_log_every == 0 or epoch == args.epochs - 1):
                fstats, prev_state = collect_strata_diagnostics(optimizer, ranks, prev_state)
                rec["flag_diag"] = fstats
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()
            if epoch % 50 == 0 or epoch == args.epochs - 1:
                print(f"[{args.method}/{args.dataset}] ep{epoch} train={train_loss:.4f} "
                      f"val={val_acc:.4f} test={test_acc:.4f} peak={peak_mem:.2f}GB t={ep_time:.1f}s", flush=True)

    total_time = time.time() - t_start
    summary = {"method": args.method, "dataset": args.dataset, "epochs_run": args.epochs,
               "final_val_acc": val_acc, "final_test_acc": test_acc, "best_val_acc": best_val,
               "total_time_s": total_time, "avg_epoch_time_s": total_time / args.epochs,
               "peak_mem_gb": peak_mem, "state_budget": budget,
               "timestamp_end": datetime.now(JST).isoformat()}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {args.method}/{args.dataset}: test={test_acc:.4f} "
          f"avg_ep={total_time/args.epochs:.1f}s total={total_time/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
