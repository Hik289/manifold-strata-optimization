"""UCI data; train-only preprocessing, deterministic frozen held-out splits."""
import argparse,csv,gzip,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import requests
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)/'tabular';root.mkdir(parents=True,exist_ok=True)

def get(number,name):
 f=root/(name+'.zip')
 if not f.exists():
  url=f'https://archive.ics.uci.edu/static/public/{number}/{name.replace("_","+")}.zip';r=requests.get(url,timeout=(20,120));r.raise_for_status();f.write_bytes(r.content)
 return zipfile.ZipFile(f),hashlib.sha256(f.read_bytes()).hexdigest()
def split_ids(keys):
 h=np.array([int(hashlib.sha256(('seed42/'+k).encode()).hexdigest()[:8],16)%100 for k in keys]);return {'train':np.where(h<70)[0],'validation':np.where((h>=70)&(h<85))[0],'test':np.where(h>=85)[0]}
def save(name,x,y,ids,meta):
 tr=ids['train'];mean=x[tr].mean(0);sd=x[tr].std(0);sd=np.where(sd<1e-6,1.,sd);x=((x-mean)/sd).astype('float32');assert np.isfinite(x).all();n=max(y)+1
 values={f'{s}_{k}':v[ix] for s,ix in ids.items() for k,v in [('x',x),('y',y)]};np.savez_compressed(root/(name+'.npz'),**values,mean=mean,scale=sd)
 meta.update(seed=42,n_classes=int(n),n_features=x.shape[1],counts={s:len(ix) for s,ix in ids.items()},preprocessing='fit means/scales and categorical vocabularies on training only; preserve all records',sha256=hashlib.sha256((root/(name+'.npz')).read_bytes()).hexdigest())
 (root/(name+'.json')).write_text(json.dumps(meta,indent=2));print(json.dumps(dict(dataset=name,**meta)),flush=True)

z,digest=get(2,'adult')
def adult_rows(name):
 return [[v.strip() for v in row] for row in csv.reader(io.StringIO(z.read(name).decode())) if len(row)==15]
tr=adult_rows('adult.data');te=adult_rows('adult.test');rows=tr+te;valmask=np.array([int(hashlib.sha256(('seed42/'+str(i)).encode()).hexdigest()[:8],16)%10==0 for i in range(len(tr))]);ids={'train':np.where(~valmask)[0],'validation':np.where(valmask)[0],'test':np.arange(len(tr),len(rows))};numeric={0,2,4,10,11,12};cols=[]
for j in range(14):
 if j in numeric:cols.append(np.array([float(r[j]) for r in rows])[:,None]);continue
 vocab=sorted({rows[i][j] for i in ids['train']});index={v:i for i,v in enumerate(vocab)};v=np.zeros((len(rows),len(vocab)+1),dtype=np.float32)
 for i,row in enumerate(rows):v[i,index.get(row[j],len(vocab))]=1
 cols.append(v)
x=np.concatenate(cols,1);y=np.array([int(r[-1].rstrip('.')=='>50K') for r in rows],dtype=np.int64)
assert len(rows)==48842
save('adult',x,y,ids,dict(source='https://archive.ics.uci.edu/dataset/2/adult',archive_sha256=digest,protocol='official test untouched; hash 10% official train as internal validation; unknown categories get explicit train-fitted UNK column'))
z,digest=get(186,'wine_quality')
for color in ['red','white']:
 raw=z.read(f'winequality-{color}.csv').decode();data=np.loadtxt(io.StringIO(raw),delimiter=';',skiprows=1);x=data[:,:-1];y=(data[:,-1]>=6).astype(np.int64);keys=[','.join(map(str,row)) for row in data];ids=split_ids(keys)
 save('wine_'+color,x,y,ids,dict(source='https://archive.ics.uci.edu/dataset/186/wine+quality',archive_sha256=digest,protocol='quality>=6 binary target; hash-identical full rows share split, 70/15/15; red/white separate tasks'))
z,digest=get(31,'covertype');name=next(n for n in z.namelist() if n.endswith('covtype.data.gz'));data=np.loadtxt(io.BytesIO(gzip.decompress(z.read(name))),delimiter=',');x=data[:,:-1];y=data[:,-1].astype(np.int64)-1;assert len(x)==581012
ids=split_ids([','.join(map(str,row)) for row in data]);save('covertype',x,y,ids,dict(source='https://archive.ics.uci.edu/dataset/31/covertype',archive_sha256=digest,protocol='full 581012 rows; hash-identical rows share 70/15/15 split'))
