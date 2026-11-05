"""Download a prepared lightweight snapshot from an explicitly configured SSH host."""
import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Your SSH alias")
    parser.add_argument("--remote-root", required=True, help="Project directory on that host")
    parser.add_argument("--jump", help="Optional SSH jump alias")
    parser.add_argument("--destination", type=Path, default=Path("snapshot_light"))
    args = parser.parse_args()
    transport = ["ssh"] + (["-J", args.jump] if args.jump else [])
    args.destination.mkdir(parents=True, exist_ok=True)
    remote = args.host + ":" + args.remote_root.rstrip("/") + "/snapshot_light/"
    subprocess.run(["rsync", "-a", "--protect-args", "-e", shlex.join(transport),
                    remote, str(args.destination) + "/"], check=True)
    print("Snapshot downloaded. The active experiment queue has not been replaced.")


if __name__ == "__main__":
    main()
