"""Initialize a local queue from explicit recipe selections, without overwriting a queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family")
    parser.add_argument("--out-prefix", default="")
    parser.add_argument("--ids", help="Comma-separated recipe IDs")
    parser.add_argument("--all", action="store_true", help="Explicitly select the full design")
    parser.add_argument("--queue", type=Path, default=root / "experiments/revision/queue.json")
    args = parser.parse_args()
    if not (args.family or args.ids or args.all):
        parser.error("Select --family, --ids, or --all")
    ids = {int(x) for x in args.ids.split(",")} if args.ids else None
    recipes = json.loads((root / "configs/experiments.json").read_text())
    selected = [j for j in recipes if (not args.family or j["family"] == args.family)
                and (ids is None or j["id"] in ids) and j["out"].startswith(args.out_prefix)]
    if not selected or (ids is not None and {j["id"] for j in selected} != ids):
        parser.error("Selection is empty or contains unknown/conflicting IDs")
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    with args.queue.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.queue.exists():
            parser.error("Queue already exists; use a fresh --queue path")
        jobs = []
        for source in selected:
            job = dict(source)
            job["requires"] = [str(root / "data" / p) for p in source["requires"]]
            job["status"] = "pending" if all(Path(p).exists() for p in job["requires"]) else "waiting_data"
            jobs.append(job)
        temporary = args.queue.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(jobs, indent=2))
        temporary.replace(args.queue)
    print(f"Initialized {len(jobs)} local recipes; data-ready: {sum(j['status'] == 'pending' for j in jobs)}")


if __name__ == "__main__":
    main()
