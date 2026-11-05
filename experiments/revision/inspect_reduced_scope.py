"""Read the revised experimental scope under the authoritative queue lock."""
import argparse
import collections
import fcntl
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--queue', type=Path, default=Path('revision/queue.json'))
parser.add_argument('--manifest', type=Path, default=Path('revision/scope_20261104_manifest.json'))
args = parser.parse_args()
with args.queue.with_suffix('.lock').open() as lock:
    fcntl.flock(lock, fcntl.LOCK_SH)
    jobs = json.loads(args.queue.read_text())
manifest = json.loads(args.manifest.read_text())
by_id = {job['id']: job for job in jobs}
cohorts = []
for cohort in manifest['cohorts']:
    members = [by_id[index] for index in cohort['ids']]
    counts = dict(collections.Counter(job['status'] for job in members))
    cohorts.append({
        **cohort,
        'counts': counts,
        'all_prescribed_candidates_complete': all(job['status'] == 'complete' for job in members),
        'remaining_ids': [job['id'] for job in members if job['status'] != 'complete'],
    })
counts = dict(collections.Counter(job['status'] for job in jobs))
print(json.dumps({
    'counts': counts,
    'remaining_retained': counts.get('pending', 0) + counts.get('running', 0),
    'deferred_are_not_completed': True,
    'cohorts': cohorts,
    'failed_ids': [job['id'] for job in jobs if job['status'] == 'failed'],
}, indent=2))
