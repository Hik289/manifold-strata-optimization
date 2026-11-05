"""
STRATA anchor_3: 五类任务数据管线校验 (ASSERT)
逐类检查 shape / range / 统计量 / split 无泄漏 是否符合 spec。
所有结果打印为 JSON, 供 eda.md 汇总。seed=42。
运行: conda activate strata; python validate_pipelines.py
"""
from __future__ import annotations
from pathlib import Path
import json, hashlib, sys, os
import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
RESULTS = {}
DATA_ROOT = str(Path(__file__).resolve().parents[2] / 'data/raw')
os.makedirs(DATA_ROOT, exist_ok=True)


def rec(name, ok, **kw):
    d = {"pass": bool(ok)}; d.update(kw); RESULTS[name] = d
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {json.dumps(kw, default=str)[:400]}")


# ---------------- 1. LANGUAGE: WikiText-103 ----------------
def check_lm():
    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        splits = {}
        idset = {}
        for sp in ["train", "validation", "test"]:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=sp)
            # 拼接前 N 段做 tokenize 抽样统计 (完整 tokenize train 太慢, 抽样)
            n_probe = min(len(ds), 2000)
            txt = "\n".join(ds[i]["text"] for i in range(n_probe))
            ids = tok(txt)["input_ids"]
            splits[sp] = {
                "rows": len(ds),
                "probe_rows": n_probe,
                "probe_tokens": len(ids),
                "tok_min": int(min(ids)), "tok_max": int(max(ids)),
                "vocab_size": tok.vocab_size,
            }
            # 用于泄漏检查: 只统计"实质内容行"(长度>200字符, 排除空行/章节标题样板)
            def is_content(s):
                s = s.strip()
                return len(s) > 200 and not (s.startswith("=") and s.endswith("="))
            idset[sp] = set(
                hashlib.md5(ds[i]["text"].strip().encode()).hexdigest()
                for i in range(len(ds)) if is_content(ds[i]["text"])
            )
        # split 泄漏: 训练/验证/测试实质内容段重叠比例
        tr, va, te = idset["train"], idset["validation"], idset["test"]
        leak_va = len(tr & va) / max(len(va), 1)
        leak_te = len(tr & te) / max(len(te), 1)
        # token id 必须 < vocab_size
        rng_ok = all(splits[sp]["tok_max"] < splits[sp]["vocab_size"] for sp in splits)
        # 泄漏容忍: WikiText 有大量重复空行/标题行, 只看实质重叠; 阈值 5%
        leak_ok = leak_va < 0.05 and leak_te < 0.05
        rec("LM_wikitext103", rng_ok and leak_ok,
            splits=splits, leak_val=round(leak_va, 5), leak_test=round(leak_te, 5),
            note="GPT-2 BPE tokenizer; token id range & split leakage checked (probe-tokenized).")
    except Exception as e:
        rec("LM_wikitext103", False, error=repr(e))


# ---------------- 2. VISION: CIFAR-10 / CIFAR-100 ----------------
def _load_cifar_hf(repo, ncls):
    """用 HF datasets 加载 CIFAR (torchvision toronto 镜像被限速)。返回 train/test 的 X(N,32,32,3) float[0,1], y。"""
    from datasets import load_dataset
    img_key = "img" if repo.endswith("cifar10") else "img"
    lab_key = "label" if repo.endswith("cifar10") else "fine_label"
    out = {}
    for sp in ["train", "test"]:
        ds = load_dataset(repo, split=sp)
        keys = ds.column_names
        ik = "img" if "img" in keys else ("image" if "image" in keys else keys[0])
        lk = "label" if "label" in keys else ("fine_label" if "fine_label" in keys else keys[-1])
        X = np.stack([np.array(im) for im in ds[ik]]).astype(np.float32) / 255.0
        y = np.array(ds[lk])
        out[sp] = (X, y)
    return out


def check_vision():
    try:
        out = {}
        for name, repo, ncls in [("CIFAR10", "uoft-cs/cifar10", 10),
                                  ("CIFAR100", "uoft-cs/cifar100", 100)]:
            d = _load_cifar_hf(repo, ncls)
            Xtr, ytr = d["train"]; Xte, yte = d["test"]
            # float64 累加, 否则 5e7 元素上 float32 mean 会灾难性精度损失 (饱和到 ~0.328)
            mean = Xtr.reshape(-1, 3).mean(0, dtype=np.float64).tolist()
            std = Xtr.reshape(-1, 3).std(0, dtype=np.float64).tolist()
            cnt = np.bincount(ytr, minlength=ncls)
            def hset(arr, k=5000):
                idx = np.linspace(0, len(arr) - 1, min(k, len(arr))).astype(int)
                return set(hashlib.md5(arr[i].tobytes()).hexdigest() for i in idx)
            leak = len(hset((Xtr * 255).astype(np.uint8)) & hset((Xte * 255).astype(np.uint8)))
            out[name] = {
                "train_N": len(Xtr), "test_N": len(Xte),
                "shape": list(Xtr.shape[1:]), "range": [float(Xtr.min()), float(Xtr.max())],
                "n_classes": ncls, "class_min": int(cnt.min()), "class_max": int(cnt.max()),
                "chan_mean": [round(m, 4) for m in mean], "chan_std": [round(s, 4) for s in std],
                "train_test_leak_count": int(leak),
            }
        ok = all(o["train_test_leak_count"] == 0 and o["range"][1] <= 1.0 for o in out.values())
        rec("VISION_cifar", ok, datasets=out, note="HF uoft-cs/cifar; uint8->[0,1]; per-channel mean/std; class balance; byte-hash leak check.")
    except Exception as e:
        rec("VISION_cifar", False, error=repr(e))


# ---------------- 3. CONTINUAL LEARNING: Split CIFAR-100 + Rotated MNIST ----------------
def check_cl():
    try:
        from datasets import load_dataset
        import torchvision.transforms.functional as TF
        # --- Split CIFAR-100: 10 tasks x 10 classes (HF) ---
        ds = load_dataset("uoft-cs/cifar100", split="train")
        lk = "fine_label" if "fine_label" in ds.column_names else "label"
        y = np.array(ds[lk])
        n_tasks = 10
        task_classes = [list(range(t * 10, t * 10 + 10)) for t in range(n_tasks)]
        allc = sorted(sum(task_classes, []))
        disjoint = (allc == list(range(100)))
        task_sizes = [int(np.isin(y, tc).sum()) for tc in task_classes]
        split_cifar = {"n_tasks": n_tasks, "classes_per_task": 10,
                       "disjoint_and_complete": disjoint,
                       "task_train_sizes": task_sizes}
        # --- Rotated MNIST: 逐 task 旋转角度 (HF) ---
        mds = load_dataset("ylecun/mnist", split="train")
        ik = "image" if "image" in mds.column_names else "img"
        img0 = torch.tensor(np.array(mds[0][ik]), dtype=torch.float32) / 255.0
        angles = list(range(0, 180, 20))  # 9 tasks 示例
        rot_ranges = []
        for a in angles:
            r = TF.rotate(img0.unsqueeze(0), a).squeeze(0)
            rot_ranges.append([float(r.min()), float(r.max())])
        rot_mnist = {"n_tasks": len(angles), "angles": angles,
                     "img_range_ok": all(0 <= lo and hi <= 1.0 + 1e-4 for lo, hi in rot_ranges),
                     "base_shape": list(img0.shape)}
        ok = disjoint and rot_mnist["img_range_ok"] and min(task_sizes) == max(task_sizes)
        rec("CL_split_cifar_rot_mnist", ok, split_cifar100=split_cifar, rotated_mnist=rot_mnist,
            note="task class disjointness/completeness; balanced task sizes; rotation preserves range.")
    except Exception as e:
        rec("CL_split_cifar_rot_mnist", False, error=repr(e))


# ---------------- 4. GRAPH: Planetoid (Cora/Citeseer/PubMed) + ogbn-arxiv ----------------
def check_graph():
    out = {}
    try:
        from torch_geometric.datasets import Planetoid
        for name in ["Cora", "Citeseer", "PubMed"]:
            ds = Planetoid(root=os.path.join(DATA_ROOT, "Planetoid"), name=name)
            g = ds[0]
            # split mask 无重叠
            tr, va, te = g.train_mask, g.val_mask, g.test_mask
            overlap = int((tr & va).sum() + (tr & te).sum() + (va & te).sum())
            out[name] = {
                "n_nodes": int(g.num_nodes), "n_edges": int(g.num_edges),
                "n_feats": int(g.num_node_features), "n_classes": int(ds.num_classes),
                "feat_range": [float(g.x.min()), float(g.x.max())],
                "label_range": [int(g.y.min()), int(g.y.max())],
                "split_sizes": [int(tr.sum()), int(va.sum()), int(te.sum())],
                "mask_overlap": overlap,
            }
    except Exception as e:
        out["Planetoid_error"] = repr(e)
    try:
        # torch>=2.6 默认 weights_only=True 会破坏 ogb 的 .pt 加载; 打补丁
        import torch as _t
        _orig_load = _t.load
        _t.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=os.path.join(DATA_ROOT, "ogb"))
        _t.load = _orig_load
        g = ds[0]
        sp = ds.get_idx_split()
        tr, va, te = set(sp["train"].tolist()), set(sp["valid"].tolist()), set(sp["test"].tolist())
        overlap = len(tr & va) + len(tr & te) + len(va & te)
        out["ogbn-arxiv"] = {
            "n_nodes": int(g.num_nodes), "n_edges": int(g.num_edges),
            "n_feats": int(g.num_node_features), "n_classes": int(ds.num_classes),
            "feat_range": [round(float(g.x.min()), 4), round(float(g.x.max()), 4)],
            "split_sizes": [len(tr), len(va), len(te)],
            "idx_overlap": overlap,
        }
    except Exception as e:
        out["ogbn-arxiv_error"] = repr(e)
    ok = all("error" not in k for k in out) and \
         all(v.get("mask_overlap", v.get("idx_overlap", 1)) == 0 for k, v in out.items() if isinstance(v, dict))
    rec("GRAPH_planetoid_ogb", ok, datasets=out, note="node feat range; label range; split mask/idx disjointness.")


# ---------------- 5. SCIML: Burgers + Poisson (合成) ----------------
def check_sciml():
    try:
        # --- 1D viscous Burgers, 稳定有限差分 (upwind 平流 + 中心扩散, 固定 Dirichlet BC) ---
        nu = 0.01 / np.pi
        nx, nt = 256, 100
        x = np.linspace(-1, 1, nx)
        dx = x[1] - x[0]
        # CFL: dt < min(dx/max|u|, dx^2/(2nu)); 取保守值
        dt = 0.2 * min(dx / 1.0, dx * dx / (2 * nu))
        u = -np.sin(np.pi * x)  # 初值
        u[0] = 0.0; u[-1] = 0.0  # Dirichlet BC
        T_final = 1.0
        steps = int(T_final / dt)
        snap_every = max(1, steps // nt)
        snaps = [u.copy()]
        for s in range(steps):
            ux_c = np.zeros_like(u)
            # upwind 平流: u>0 用后向差分, u<0 用前向差分
            up = u[1:-1] > 0
            back = (u[1:-1] - u[:-2]) / dx
            fwd = (u[2:] - u[1:-1]) / dx
            ux_up = np.where(up, back, fwd)
            uxx = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx * dx)
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + dt * (-u[1:-1] * ux_up + nu * uxx)
            u_new[0] = 0.0; u_new[-1] = 0.0
            u = u_new
            if (s + 1) % snap_every == 0:
                snaps.append(u.copy())
        U = np.array(snaps)  # (~nt+1, nx)
        burgers = {
            "shape": list(U.shape), "u_range": [float(U.min()), float(U.max())],
            "finite": bool(np.isfinite(U).all()),
            "nu": nu, "domain": "x in [-1,1], t in [0,1]",
        }
        # --- 2D Poisson -Δu=f, 制造解 u=sin(pi x)sin(pi y) ---
        n = 64
        xs = np.linspace(0, 1, n)
        XX, YY = np.meshgrid(xs, xs)
        u_true = np.sin(np.pi * XX) * np.sin(np.pi * YY)
        f = 2 * np.pi ** 2 * np.sin(np.pi * XX) * np.sin(np.pi * YY)  # -Δu = f
        poisson = {
            "shape": list(u_true.shape), "u_range": [float(u_true.min()), float(u_true.max())],
            "f_range": [float(f.min()), float(f.max())],
            "bc_zero": bool(np.allclose(u_true[0, :], 0) and np.allclose(u_true[-1, :], 0)
                            and np.allclose(u_true[:, 0], 0) and np.allclose(u_true[:, -1], 0)),
        }
        ok = burgers["finite"] and poisson["bc_zero"] and abs(burgers["u_range"][0]) <= 1.5
        rec("SCIML_burgers_poisson", ok, burgers=burgers, poisson=poisson,
            note="Burgers FD reference (nu=0.01/pi); Poisson manufactured solution w/ zero BC.")
    except Exception as e:
        rec("SCIML_burgers_poisson", False, error=repr(e))


if __name__ == "__main__":
    print("=" * 60, "\nSTRATA anchor_3 数据管线校验 (seed=42)\n", "=" * 60)
    check_lm()
    check_vision()
    check_cl()
    check_graph()
    check_sciml()
    n_pass = sum(v["pass"] for v in RESULTS.values())
    RESULTS["_summary"] = {"n_checks": len(RESULTS), "n_pass": n_pass,
                           "all_pass": n_pass == len(RESULTS)}
    out_path = str(Path(__file__).resolve().parents[2] / 'data/pipeline_validation.json')
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("\n" + "=" * 60)
    print(f"总计 {n_pass}/{len(RESULTS) - 1} 通过 -> {out_path}")
