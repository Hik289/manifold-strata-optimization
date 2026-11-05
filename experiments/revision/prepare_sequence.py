"""Freeze WikiText-2 official splits; word vocabulary is fitted on train only."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import requests
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)/'sequence';root.mkdir(parents=True,exist_ok=True);texts={};hashes={}
for s in ['train','validation','test']:
 f=root/f'wikitext2_{s}.parquet';url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/{s}-00000-of-00001.parquet'
 if not f.exists():
  r=requests.get(url,timeout=(20,180));r.raise_for_status();f.write_bytes(r.content)
 hashes[s]=hashlib.sha256(f.read_bytes()).hexdigest();texts[s]=pq.read_table(f)['text'].to_pylist()
words={s:[w for line in lines for w in (line.split()+['<eos>'])] for s,lines in texts.items()};counts=collections.Counter(words['train']);vocab=['<unk>','<eos>']+[w for w,c in sorted(counts.items(),key=lambda z:(-z[1],z[0])) if c>=2 and w not in ['<unk>','<eos>']][:29998];index={w:i for i,w in enumerate(vocab)};tokens={s:np.array([index.get(w,0) for w in ws],dtype=np.int32) for s,ws in words.items()};np.savez_compressed(root/'wikitext2_words.npz',**tokens);(root/'wikitext2_vocabulary.json').write_text(json.dumps(vocab))
meta=dict(ready=True,source='https://huggingface.co/datasets/Salesforce/wikitext',revision='b08601e04326c79dfdd32d625aee71d232d685c3',split='official wikitext-2-raw-v1 train/validation/test',tokenization='whitespace plus line EOS; train-only minfreq2, maximum30000 words',vocabulary=len(vocab),split_tokens={s:len(v) for s,v in tokens.items()},parquet_sha256=hashes,sha256=hashlib.sha256((root/'wikitext2_words.npz').read_bytes()).hexdigest());(root/'wikitext2_words.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
