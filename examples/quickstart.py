"""A data-free CPU example; this is an installation check, not a benchmark."""
import argparse
import torch
from strata_pkg import STRATAAdamW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    x = torch.randn(128, 64, device=args.device)
    target = x @ torch.randn(64, 8, device=args.device)
    model = torch.nn.Sequential(torch.nn.Linear(64, 96), torch.nn.GELU(),
                                torch.nn.Linear(96, 8)).to(args.device)
    optimizer = STRATAAdamW(model.parameters(), lr=1e-3, ranks=[4, 8, 16],
                           rhos=[0.999, 0.98, 0.9], a_ks=[1.0, 0.7, 0.4],
                           a_perp=0.2, tau=5, xi=1e-3, eps=1e-6, seed=42)
    first = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        if first is None:
            first = loss.item()
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Non-finite optimization state")
    print(f"CPU/CUDA smoke check: {args.steps} updates, loss {first:.6f} -> {loss.item():.6f}")


if __name__ == "__main__":
    main()
