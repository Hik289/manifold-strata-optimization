"""Freeze public text subsets using document hashes, never benchmark outcomes."""
import argparse,hashlib,json
from pathlib import Path
from datasets import load_dataset
p=argparse.ArgumentParser();p.add_argument('--corpus',choices=['openwebtext','c4'],required=True);p.add_argument('--root',required=True);p.add_argument('--documents',type=int,default=100000);a=p.parse_args()
root=Path(a.root)/a.corpus;root.mkdir(parents=True,exist_ok=True)
source='Skylion007/openwebtext' if a.corpus=='openwebtext' else 'allenai/c4'
args=[source]+(['en'] if a.corpus=='c4' else [])
stream=load_dataset(*args,split='train',streaming=True)
files={s:(root/f'{s}.jsonl.partial').open('w') for s in ['train','validation','test']};counts={s:0 for s in files};seen=set();digests={s:hashlib.sha256() for s in files}
try:
 for row in stream:
  text=row['text'];digest=hashlib.sha256(text.encode()).hexdigest()
  if digest in seen:continue
  seen.add(digest);bucket=int(digest[:8],16)%100;split='test' if bucket==0 else 'validation' if bucket==1 else 'train'
  line=json.dumps(dict(text=text,sha256=digest),ensure_ascii=False)+'\n';files[split].write(line);digests[split].update(line.encode());counts[split]+=1
  if len(seen)>=a.documents:break
finally:
 for f in files.values():f.close()
if len(seen)<a.documents:raise RuntimeError('Incomplete corpus stream; partial files preserved')
for s in files:(root/f'{s}.jsonl.partial').replace(root/f'{s}.jsonl')
(root/'provenance.json').write_text(json.dumps(dict(source=source,documents=len(seen),split_rule='sha256(document) modulo 100: 0=test,1=validation,else=train',split_counts=counts,split_sha256={s:d.hexdigest() for s,d in digests.items()}),indent=2))
