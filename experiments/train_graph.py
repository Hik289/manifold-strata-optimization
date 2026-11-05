"""STRATA wave2 图任务训练: ogbn-arxiv / GraphSAGE, 全批节点分类。

复用 train.py 的 optimizer builder + collect_strata_diagnostics (全轨迹 δ_nest)。
judge 同 wave1: 末端 100ep 均值 STRATA vs Single-Grassmann vs GaLore vs AdamW。
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
from data_graph import load_arxiv  # noqa: E402
from model_graph import GraphSAGE, count_params  # noqa: E402
from state_budget import budget_report  # noqa: E402
from train import build_optimizer, collect_strata_diagnostics  # noqa: E402

JST = timezone(timedelta(hours=9))


@torch.no_grad()
def evaluate(model, data, idx, criterion):
    model.eval()
    out = model(data["x"], data["adj_idx"], data["adj_val"], data["n"])
    logits = out[idx]
    y = data["y"][idx]
    loss = criterion(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["adamw", "galore", "single_grassmann",
                             "independent_grassmann", "strata"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--xi", type=float, default=1e-3)
    ap.add_argument("--tau", type=int, default=50)
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--a_ks", type=float, nargs="*", default=None)
    ap.add_argument("--a_perp", type=float, default=0.2)
    ap.add_argument("--galore_tau", type=int, default=200)
    ap.add_argument("--galore_scale", type=float, default=0.25)
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

    data = load_arxiv(args.data_dir, device=device)
    model = GraphSAGE(data["feat_dim"], args.hidden, data["n_cls"],
                      args.n_layers, args.dropout).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer, opt_meta = build_optimizer(args.method, model, args)

    ranks = [args.R // 4, args.R // 2, args.R]
    K = len(ranks)
    budget = budget_report(model, args.R, ranks, K, args.p)

    def lr_at(epoch):
        if epoch < args.warmup:
            return args.lr * (epoch + 1) / args.warmup
        prog = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))

    meta = {
        "project_id": "manifoldstrata", "phase": "RUNNING-wave2",
        "method": args.method, "task": "ogbn-arxiv / GraphSAGE",
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
            optimizer.zero_grad(set_to_none=True)
            out = model(data["x"], data["adj_idx"], data["adj_val"], data["n"])
            loss = criterion(out[data["train_idx"]], data["y"][data["train_idx"]])
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            optimizer.step()
            ep_time = time.time() - ep_t0
            peak_mem = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0.0

            val_loss, val_acc = evaluate(model, data, data["val_idx"], criterion)
            test_loss, test_acc = evaluate(model, data, data["test_idx"], criterion)
            best_val = max(best_val, val_acc)

            rec = {
                "epoch": epoch, "lr": lr, "train_loss": float(loss.item()),
                "val_loss": val_loss, "val_acc": val_acc,
                "test_loss": test_loss, "test_acc": test_acc,
                "grad_norm": float(gn), "epoch_time_s": ep_time, "peak_mem_gb": peak_mem,
            }
            if is_flaglike and (epoch % args.flag_log_every == 0 or epoch == args.epochs - 1):
                fstats, prev_state = collect_strata_diagnostics(optimizer, ranks, prev_state)
                rec["flag_diag"] = fstats
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logf.flush()

            if epoch % 100 == 0 or epoch == args.epochs - 1:
                print(f"[{args.method}] ep{epoch} train={loss.item():.4f} "
                      f"val={val_acc:.4f} test={test_acc:.4f} peak={peak_mem:.2f}GB "
                      f"t={ep_time*1000:.0f}ms", flush=True)

    total_time = time.time() - t_start
    summary = {
        "method": args.method, "epochs_run": args.epochs,
        "final_val_acc": val_acc, "final_test_acc": test_acc,
        "best_val_acc": best_val, "total_time_s": total_time,
        "avg_epoch_time_s": total_time / args.epochs,
        "peak_mem_gb": peak_mem, "state_budget": budget,
        "timestamp_end": datetime.now(JST).isoformat(),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {args.method}: final_test={test_acc:.4f} "
          f"avg_epoch={total_time/args.epochs*1000:.0f}ms total={total_time/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
