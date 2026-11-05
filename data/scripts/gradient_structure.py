"""
STRATA 核心经验前置诊断 (H0 旗结构是否有经验基础)。

问题: 深度学习梯度协方差在不同记忆时间尺度 (ρ_1>...>ρ_K) 下,
其主子空间是否近似嵌套, 且内层(长时间尺度)是否比外层(短时间尺度)更稳定?

在一个可完整计算协方差的小模型 + 小任务上 (无 sketch, 精确 eigh):
  C_{k,t} = ρ_k C_{k,t-1} + (1-ρ_k) G_t^T G_t   (README §2.1 / §3.5)
  V̂_{k,t} = Top_{r_k}(C_{k,t})                    (README §2.3 H1)

量化判据:
  1. 嵌套缺口 δ_nest = Σ ||(I-P̂_{k+1})P̂_k||_F^2 (归一化) vs 随机零假设
  2. 时间尺度间主子空间重叠 (同秩主角) + 交换性 proxy Comm(C_i,C_j)
  3. 各层漂移速度 d_k,t = ||P_k,t - P_k,t-1||_F, 内层 vs 外层 (H2)
  4. 各时间尺度谱 (spike-tail, 对照 Spectra ~1.5% / k*≈39)

结论: 支持 / 削弱 旗结构假设。seed=42。
"""
from __future__ import annotations
from pathlib import Path
import json, os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from strata_pkg import diagnostics as diag  # 复用 ML Engineer 的定义 (一致性)

SEED = 42
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = str(Path(__file__).resolve().parents[2] / 'data/raw')


def set_seed(s=SEED):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------- 随机嵌套子空间零假设 δ_nest ----------
def random_nesting_gap(n, ranks, trials=200):
    """随机 (Haar) 嵌套基 V_1⊂...⊂V_K 的 δ_nest 期望 (零假设标尺)。
    注意: 若强制嵌套采样则 gap=0; 这里采样 K 个 *独立* 随机子空间,
    衡量'若各时间尺度主子空间无关联时' δ_nest 的量级。"""
    vals = []
    for _ in range(trials):
        bases = []
        for r in ranks:
            Z = torch.randn(n, r)
            Q, _ = torch.linalg.qr(Z)
            bases.append(Q[:, :r])
        vals.append(diag.nesting_gap(bases))
    return float(np.mean(vals)), float(np.std(vals))


def norm_nesting_gap(bases, ranks):
    """δ_nest 及归一化版 (除以 Σ_{k<K} r_k = 完全正交时的最大值)。"""
    raw = diag.nesting_gap(bases)
    maxv = float(sum(ranks[:-1]))  # 每层 (I-P_{k+1})P_k 最大 Frobenius^2 = r_k
    return raw, raw / (maxv + 1e-12)


# ---------- 同秩子空间重叠 (时间尺度间共享方向) ----------
def subspace_overlap_same_rank(C_list, r):
    """所有时间尺度取 top-r, 两两平均主角 cos^2 (1=完全重叠,0=正交)。"""
    K = len(C_list)
    Vs = [diag.top_r_subspace(C, r) for C in C_list]
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ang = diag.principal_angles(Vs[i], Vs[j])
            M[i, j] = float(torch.cos(ang).pow(2).mean())
    return M


def spectrum_stats(C, top=None):
    evals = torch.linalg.eigvalsh(C).flip(0).clamp(min=0)  # 降序
    tot = float(evals.sum()) + 1e-12
    frac = (evals / tot).cpu().numpy()
    cum = np.cumsum(frac)
    # 达到 90% 能量所需维数; top-1.5% 维数占能量; 有效维数 tr(C)/λ_max
    n = len(frac)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k_eff = float(tot / (float(evals.max()) + 1e-12))
    frac_top1p5 = float(cum[max(0, int(0.015 * n))])
    return {"k90": k90, "eff_dim": round(k_eff, 2),
            "top1.5pct_energy": round(frac_top1p5, 4),
            "lambda_max_frac": round(float(frac[0]), 4)}


# ===================== 模型 & 数据 =====================
class SmallMLP(nn.Module):
    def __init__(self, d_in=3072, h1=256, h2=128, n_cls=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_cls)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_cifar_loader(batch=128):
    # 用 HF datasets (torchvision 的 toronto 镜像被限速)
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", split="train")
    imgs = np.stack([np.array(im) for im in ds["img"]]).astype(np.float32) / 255.0
    X = torch.tensor(imgs).reshape(len(ds), -1)
    mean = X.mean(0, keepdim=True); std = X.std(0, keepdim=True) + 1e-6
    X = (X - mean) / std
    y = torch.tensor(ds["label"], dtype=torch.long)
    return X, y, batch


class TinyTransformer(nn.Module):
    """字符级 tiny transformer, 用于跨域证据。"""
    def __init__(self, vocab, d=128, nhead=4, seqlen=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seqlen, d))
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=256,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d, vocab)
        self.seqlen = seqlen

    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.size(1)]
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        h = self.enc(h, mask=mask)
        return self.head(h)


def get_text_data(seqlen=64):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    txt = "\n".join(ds[i]["text"] for i in range(3000))[:200000]
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
    return data, len(chars), seqlen


# ===================== 诊断主循环 =====================
def run_diagnostic(setting, steps=1500, log_every=50,
                   rhos=(0.999, 0.98, 0.9), ranks=(8, 16, 32),
                   track_layer="fc1", batch=128, lr=1e-3):
    """
    对指定 setting 训练小模型, 追踪 track_layer 权重梯度右侧协方差
    C_t = G^T G (n x n), 维护 K 个时间尺度 EMA, 精确 eigh。
    """
    set_seed()
    K = len(rhos)
    assert len(ranks) == K
    print(f"\n### setting={setting} | rhos={rhos} ranks={ranks} layer={track_layer}")

    if setting == "mlp_cifar":
        X, y, batch = get_cifar_loader(batch)
        X, y = X.to(DEV), y.to(DEV)
        model = SmallMLP().to(DEV)
        def sample():
            idx = torch.randint(0, X.size(0), (batch,), device=DEV)
            return X[idx], y[idx]
        def loss_fn(m, xb, yb):
            return F.cross_entropy(m(xb), yb)
        n_data = X.size(0)
    elif setting == "tf_text":
        data, vocab, seqlen = get_text_data()
        data = data.to(DEV)
        model = TinyTransformer(vocab, seqlen=seqlen).to(DEV)
        def sample():
            ix = torch.randint(0, data.size(0) - seqlen - 1, (batch,), device=DEV)
            xb = torch.stack([data[i:i + seqlen] for i in ix])
            yb = torch.stack([data[i + 1:i + seqlen + 1] for i in ix])
            return xb, yb
        def loss_fn(m, xb, yb):
            out = m(xb)
            return F.cross_entropy(out.reshape(-1, out.size(-1)), yb.reshape(-1))
        n_data = data.size(0)
    else:
        raise ValueError(setting)

    # 找到 track_layer 的权重, 决定 n = min(m,n) 侧协方差
    W = dict(model.named_parameters())[f"{track_layer}.weight"]  # (out, in)
    m_dim, n_dim = W.shape
    right = n_dim <= m_dim  # 在较小一侧维护协方差
    n = min(m_dim, n_dim)
    print(f"track W shape=({m_dim},{n_dim}) -> covariance side n={n} ({'right G^TG' if right else 'left GG^T'})")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    C = [torch.zeros(n, n, device=DEV) for _ in range(K)]
    prev_P = [None] * K
    logs = []

    for t in range(steps):
        xb, yb = sample()
        opt.zero_grad()
        loss = loss_fn(model, xb, yb)
        loss.backward()
        G = W.grad.detach()  # (m,n)
        Ct = (G.t() @ G) if right else (G @ G.t())  # (n,n)
        for k in range(K):
            C[k].mul_(rhos[k]).add_(Ct, alpha=(1 - rhos[k]))
        opt.step()

        if (t + 1) % log_every == 0 and t > log_every * 2:  # 跳过 EMA 未预热阶段
            # 各时间尺度 top-r_k 主子空间 (偏差修正: EMA 已预热, 直接用)
            bases = [diag.top_r_subspace(C[k], ranks[k]) for k in range(K)]
            raw_gap, norm_gap = norm_nesting_gap(bases, ranks)
            # 漂移 d_k = ||P_k - P_k_prev||_F
            drifts = []
            for k in range(K):
                Pk = diag.projection_from_basis(bases[k])
                if prev_P[k] is not None:
                    drifts.append(float((Pk - prev_P[k]).pow(2).sum().sqrt()))
                else:
                    drifts.append(float("nan"))
                prev_P[k] = Pk
            # 交换性 proxy (相邻时间尺度)
            comms = [diag.commutator_proxy(C[k], C[k + 1]) for k in range(K - 1)]
            # 同秩重叠 (以最小秩 r_1 看长/短时间尺度是否共享内核)
            overlap = subspace_overlap_same_rank(C, ranks[0])
            logs.append({
                "step": t + 1, "loss": round(float(loss), 4),
                "nest_gap_raw": round(raw_gap, 4), "nest_gap_norm": round(norm_gap, 4),
                "drift": [round(d, 5) for d in drifts],
                "commutator_adj": [round(c, 4) for c in comms],
                "overlap_r1_long_short": round(float(overlap[0, K - 1]), 4),
            })

    # ---- 汇总 (取后半段 = 训练稳定期) ----
    half = logs[len(logs) // 2:]
    def col(key, idx=None):
        vs = [l[key][idx] if idx is not None else l[key] for l in half]
        vs = [v for v in vs if not (isinstance(v, float) and math.isnan(v))]
        return vs
    mean_gap = float(np.mean(col("nest_gap_norm")))
    drift_means = [float(np.mean(col("drift", k))) for k in range(K)]
    comm_means = [float(np.mean(col("commutator_adj", k))) for k in range(K - 1)]
    overlap_mean = float(np.mean(col("overlap_r1_long_short")))
    # 随机零假设
    rnd_raw, rnd_std = random_nesting_gap(n, ranks)
    rnd_norm = rnd_raw / (sum(ranks[:-1]) + 1e-12)
    # 谱
    spec = {f"C{k}(rho={rhos[k]})": spectrum_stats(C[k]) for k in range(K)}
    # 内层 vs 外层漂移是否单调递增 (H2)
    h2_monotonic = all(drift_means[k] <= drift_means[k + 1] + 1e-9 for k in range(K - 1))
    drift_ratio_outer_inner = drift_means[-1] / (drift_means[0] + 1e-12)

    summary = {
        "setting": setting, "n_cov": n, "rhos": list(rhos), "ranks": list(ranks),
        "track_layer": track_layer, "steps": steps, "n_logs": len(logs),
        "nest_gap_norm_mean_2ndhalf": round(mean_gap, 4),
        "nest_gap_norm_random_null": round(rnd_norm, 4),
        "nest_gap_vs_null_ratio": round(mean_gap / (rnd_norm + 1e-12), 4),
        "drift_means_inner_to_outer": [round(d, 5) for d in drift_means],
        "H2_drift_monotonic_inner_slower": h2_monotonic,
        "drift_ratio_outer_over_inner": round(drift_ratio_outer_inner, 3),
        "commutator_adjacent_mean": [round(c, 4) for c in comm_means],
        "overlap_r1_long_vs_short_mean": round(overlap_mean, 4),
        "spectrum": spec,
        "final_loss": round(float(logs[-1]["loss"]), 4),
    }
    return summary, logs


def verdict(summ):
    """基于判据给出 支持/削弱 结论。"""
    v = {}
    g = summ["nest_gap_norm_mean_2ndhalf"]; gr = summ["nest_gap_vs_null_ratio"]
    # 嵌套: 归一化 gap 远小于随机零假设 -> 结构存在
    v["nesting"] = ("support" if (g < 0.5 and gr < 0.6) else
                    ("weak" if gr < 0.9 else "against"))
    # H2 漂移单调
    v["inner_more_stable"] = "support" if summ["H2_drift_monotonic_inner_slower"] else \
        ("weak" if summ["drift_ratio_outer_over_inner"] > 1.05 else "against")
    # 时间尺度共享方向: 长短时间尺度 r_1 重叠高 + 交换性小
    ov = summ["overlap_r1_long_vs_short_mean"]
    cm = float(np.mean(summ["commutator_adjacent_mean"]))
    v["shared_core"] = ("support" if (ov > 0.6 and cm < 0.3) else
                        ("weak" if ov > 0.4 else "against"))
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="mlp_cifar", choices=["mlp_cifar", "tf_text", "both"])
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    out = {"seed": SEED, "device": DEV, "results": {}}
    settings = ["mlp_cifar", "tf_text"] if args.setting == "both" else [args.setting]
    for s in settings:
        layer = "fc1" if s == "mlp_cifar" else "enc.layers.0.linear1"
        summ, logs = run_diagnostic(s, steps=args.steps, track_layer=layer)
        summ["verdict"] = verdict(summ)
        out["results"][s] = {"summary": summ, "logs": logs}
        print("\n[VERDICT]", s, json.dumps(summ["verdict"]))
        print("[SUMMARY]", json.dumps({k: summ[k] for k in
              ["nest_gap_norm_mean_2ndhalf", "nest_gap_norm_random_null",
               "nest_gap_vs_null_ratio", "drift_means_inner_to_outer",
               "H2_drift_monotonic_inner_slower", "overlap_r1_long_vs_short_mean",
               "commutator_adjacent_mean"]}, indent=2))

    outp = str(Path(__file__).resolve().parents[2] / 'data/gradient_structure.json')
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n-> {outp}")
