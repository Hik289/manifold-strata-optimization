"""Acquire official ETT, MovieLens and gas-drift data; freeze leakage checks."""
import argparse,csv,hashlib,io,json,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--data',default='../data');a=p.parse_args();root=Path(a.data)/'new_families';root.mkdir(parents=True,exist_ok=True);raw=root/'raw';raw.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(url,name):
 target=raw/name
 if not target.exists():
  for attempt in range(3):
   try:
    req=urllib.request.Request(url,headers={'User-Agent':'ManifoldSTRATA research dataset acquisition'})
    with urllib.request.urlopen(req,timeout=60) as f:blob=f.read()
    tmp=target.with_suffix(target.suffix+'.partial');tmp.write_bytes(blob);tmp.replace(target);break
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 return target
def save(name,arrays,meta):
 target=root/(name+'.npz');meta_file=root/(name+'.json')
 if target.exists() and meta_file.exists():
  old=json.loads(meta_file.read_text());assert old['array_sha256']==sha(target);stored=np.load(target);assert set(stored.files)==set(arrays) and all(np.array_equal(stored[k],v) for k,v in arrays.items()),'Prepared data are immutable; use a new versioned directory for changes';assert old['raw_sha256']==meta['raw_sha256'];print(name,'reused immutable prepared data',flush=True);return
 np.savez_compressed(target,**arrays);meta.update(array_sha256=sha(target),seed=42,prepared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));(root/(name+'.json')).write_text(json.dumps(meta,indent=2));print(name,meta.get('counts',meta.get('shape')),flush=True)
def role_hash(key):return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%100
for name in ['ETTh1','ETTh2']:
 f=fetch('https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'+name+'.csv',name+'.csv');rows=list(csv.DictReader(f.open()));cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'];x=np.array([[float(r[k]) for k in cols] for r in rows],dtype=np.float64);dates=np.array([r['date'] for r in rows],dtype='datetime64[s]');assert len(x)>=14400 and np.isfinite(x).all() and np.all(np.diff(dates)==np.timedelta64(1,'h'))
 mean=x[:8640].mean(0);scale=x[:8640].std(0);assert np.all(scale>0);z=((x-mean)/scale).astype('float32');save(name,dict(series=z,mean=mean,scale=scale),dict(source_url='https://github.com/zhouhaoyi/ETDataset',raw_sha256=sha(f),columns=cols,shape=list(x.shape),bounds=dict(train=[0,8640],validation=[8640,11520],test=[11520,14400]),normalization='first 8640 training observations only',lookback=96,horizon=24,split_protocol='fixed ETT 12/4/4 nominal 30-day months; every target horizon lies entirely inside its split; only past observations enter inputs',target_metric='MSE in training-standard-deviation units',checks=dict(hourly_contiguous=True,finite=True,target_split_overlap=False)))
for name,url in [('ml100k','https://files.grouplens.org/datasets/movielens/ml-100k.zip'),('ml1m','https://files.grouplens.org/datasets/movielens/ml-1m.zip')]:
 f=fetch(url,name+'.zip');z=zipfile.ZipFile(f)
 if name=='ml100k':
  base=np.loadtxt(io.StringIO(z.read('ml-100k/u1.base').decode()),dtype=np.int64);test=np.loadtxt(io.StringIO(z.read('ml-100k/u1.test').decode()),dtype=np.int64);mask=np.array([role_hash(f'{u}:{i}'.encode())<10 for u,i,_,_ in base]);parts=dict(train=base[~mask],validation=base[mask],test=test);allrows=np.concatenate([base,test]);split='official u1.test; validation is SHA-256 10% of official u1.base grouped by user/item'
 else:
  text=z.read('ml-1m/ratings.dat').decode();allrows=np.array([[int(v) for v in s.split('::')] for s in text.splitlines()],dtype=np.int64);order=np.lexsort((allrows[:,1],allrows[:,3],allrows[:,0]));allrows=allrows[order];parts={k:[] for k in ['train','validation','test']}
  for u in np.unique(allrows[:,0]):
   block=allrows[allrows[:,0]==u];n=len(block);a1=int(.7*n);a2=int(.85*n)
   for k,blockpart in zip(parts,[block[:a1],block[a1:a2],block[a2:]]):parts[k].append(blockpart)
  parts={k:np.concatenate(v) for k,v in parts.items()};split='within-user chronological 70/15/15 by timestamp, item ID breaks ties'
 pairs={k:{(int(u),int(i)) for u,i,_,_ in v} for k,v in parts.items()};assert not (pairs['train']&pairs['validation'] or pairs['train']&pairs['test'] or pairs['validation']&pairs['test']);assert len(allrows)==(100000 if name=='ml100k' else 1000209);arrays={};counts={};mu=float(parts['train'][:,2].mean())
 for k,v in parts.items():arrays[k+'_ui']=(v[:,:2]-1).astype('int64');arrays[k+'_y']=v[:,2].astype('float32');counts[k]=len(v)
 save(name,arrays,dict(source_url=url,raw_sha256=sha(f),counts=counts,n_users=int(allrows[:,0].max()),n_items=int(allrows[:,1].max()),rating_train_mean=mu,split_protocol=split,checks=dict(no_user_item_pair_overlap=True,finite=True),target_metric='RMSE in raw 1--5 rating units',license='GroupLens research use; acknowledge Harper and Konstan 2015; raw data not for public redistribution'))
f=fetch('https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip','gas_drift.zip');z=zipfile.ZipFile(f);xs=[];ys=[];bs=[]
for batch in range(1,11):
 name=next(n for n in z.namelist() if n.lower().endswith(f'batch{batch}.dat'));lines=z.read(name).decode().splitlines()
 for line in lines:
  s=line.split();y=int(s[0])-1;v=np.zeros(128,dtype='float32')
  for pair in s[1:]:i,value=pair.split(':');v[int(i)-1]=float(value)
  xs.append(v);ys.append(y);bs.append(batch-1)
x=np.array(xs);y=np.array(ys,dtype='int64');batch=np.array(bs,dtype='int64');assert x.shape==(13910,128) and np.isfinite(x).all() and set(y)==set(range(6));roles=np.array([role_hash(v.tobytes()+int(t).to_bytes(1,'big')) for v,t in zip(x,y)]);role=np.where(roles<70,0,np.where(roles<85,1,2)).astype('int64');train_first=x[(batch==0)&(role==0)];mu=train_first.mean(0,dtype='float64');sd=train_first.std(0,dtype='float64');sd[sd<1e-12]=1.;scaled=((x-mu)/sd).astype('float32');counts={str(b+1):{s:int(((batch==b)&(role==i)).sum()) for i,s in enumerate(['train','validation','test'])} for b in range(10)};assert all(all(v>0 for v in d.values()) for d in counts.values())
save('gas_drift',dict(x=scaled,y=y,batch=batch,role=role),dict(source_url='https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset',raw_sha256=sha(f),counts=counts,n_features=128,n_classes=6,split_protocol='global SHA-256 of float32 feature row and label, 70/15/15; identical rows retain the same role across all batches',normalization='batch1 training rows only; no statistics fitted on later domains',protocol='10 chronological domains, 100 full epochs each; no future domains enter training or replay',checks=dict(finite=True,total_records=13910,identical_row_roles_consistent=True),target_metric='equal-batch mean held-out accuracy over seen domains',license='UCI research dataset; acknowledge original drift dataset paper'))
