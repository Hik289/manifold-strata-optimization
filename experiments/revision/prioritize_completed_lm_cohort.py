"""Close the completed GPT-2 Small R16 cohort before more model/rank expansion.

Only pending queue order changes. Running training is never interrupted.
"""
import fcntl,json,os,time
from pathlib import Path
root=Path('..').resolve();q=root/'experiments/revision/queue.json'
with q.with_suffix('.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads(q.read_text())
 anchor='runs_revision/wikitext103__gpt2-small__strata__R16__lr0.0003'
 assert any(j['out']==anchor and j['status']=='complete' for j in jobs)
 targets=[j for j in jobs if j['status']=='pending' and j['family']=='language' and 'wikitext103__gpt2-small__' in j['out'] and '__R16__' in j['out']]
 targets.sort(key=lambda j:j['id']);ids={j['id'] for j in targets};fixed=[j for j in jobs if j['status']!='pending'];rest=[j for j in jobs if j['status']=='pending' and j['id'] not in ids];tmp=q.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(fixed+targets+rest,indent=2));tmp.replace(q)
 report=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reason='complete an existing matched model/rank cohort before model/rank expansion; no test-based task selection',prioritized_ids=[j['id'] for j in targets],budgets_changed=False,running_changed=False)
(root/'results_revision/evidence_extension/lm_cohort_priority.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
