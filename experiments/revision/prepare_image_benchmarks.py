"""Acquire/validate real image benchmarks and freeze seed-42 split manifests.
Existing data are read-only; new archives, extraction and manifests stay in the
STRATA revision data directory. No dataset executable or pickle is executed.
"""
import argparse,collections,hashlib,io,json,os,re,time,zipfile
from pathlib import Path
import requests
from PIL import Image
DOMAINS=['clipart','infograph','painting','quickdraw','real','sketch']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def download(urls,dest):
 dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():return dest
 tmp=dest.with_suffix(dest.suffix+'.partial')
 for url in urls:
  for trial in range(3):
   try:
    size=tmp.stat().st_size if tmp.exists() else 0
    with requests.get(url,stream=True,timeout=(20,60),headers={'Range':f'bytes={size}-'} if size else {}) as r:
     r.raise_for_status();append=size and r.status_code==206
     with tmp.open('ab' if append else 'wb') as f:
      n=size if append else 0;last=time.monotonic()
      for chunk in r.iter_content(4<<20):
       if chunk:f.write(chunk);n+=len(chunk)
       if time.monotonic()-last>20:print(json.dumps(dict(download=url,bytes=n)),flush=True);last=time.monotonic()
    tmp.replace(dest);print(json.dumps(dict(download_complete=str(dest),bytes=dest.stat().st_size)),flush=True);return dest
   except Exception as e:print(json.dumps(dict(url=url,attempt=trial,error=str(e))),flush=True);time.sleep(2)
 raise RuntimeError(f'All download attempts failed for {dest}')
def extract_zip(archive,root):
 root=Path(root);mark=root/(Path(archive).name+'.extracted.json')
 if mark.exists():return
 root.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  for n in z.namelist():
   if not (root/n).resolve().is_relative_to(root.resolve()):raise ValueError(f'Unsafe archive member {n}')
  bad=z.testzip()
  if bad:raise ValueError(f'CRC failure: {bad}')
  z.extractall(root)
 mark.write_text(json.dumps(dict(archive=str(archive),sha256=sha(archive),crc_ok=True)))
def bucket(key):return int(hashlib.sha256(('seed42/'+key).encode()).hexdigest()[:8],16)%10
def publish(root,rows,classes,source,protocol,expected=None):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);counts=collections.Counter(r['split'] for r in rows);seen=set();bytask=collections.Counter();byclass=collections.Counter()
 for r in rows:
  f=Path(r['path'].split('#')[0]);assert (f.is_dir() if r.get('storage')=='hf_arrow' else f.is_file()),f
  key=(r['path'],r['split']);assert key not in seen,key;seen.add(key);bytask[(r['task'],r['split'])]+=1;byclass[(r['label'],r['split'])]+=1
 assert all(counts[s]>0 for s in ['train','validation','test'])
 paths={s:{r['path'] for r in rows if r['split']==s} for s in counts};assert not (paths['train']&paths['validation'] or paths['train']&paths['test'] or paths['validation']&paths['test'])
 if expected:assert len(rows)==expected,(len(rows),expected)
 # Decode at least one image from every class/split, not just file existence.
 examples={}
 for r in rows:examples.setdefault((r['label'],r['split']),r)
 arrow_cache={}
 for r in examples.values():
  if r.get('storage')=='hf_arrow':
   from datasets import Dataset,concatenate_datasets,Image as HFImage
   cache,ref=r['path'].split('#');official,index=ref.split('/');key=(cache,official)
   if key not in arrow_cache:arrow_cache[key]=concatenate_datasets([Dataset.from_file(str(f)) for f in sorted(Path(cache).glob(f'imagenet-100-{official}*.arrow'))]).cast_column('image',HFImage(decode=False))
   picture=arrow_cache[key][int(index)]['image'];im=Image.open(io.BytesIO(picture['bytes'])) if picture.get('bytes') is not None else Image.open(picture['path'])
  else:im=Image.open(r['path'])
  with im:im.convert('RGB').load()
 tmp=root/'manifest.jsonl.partial'
 with tmp.open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 tmp.replace(root/'manifest.jsonl');info=dict(ready=True,n_classes=len(classes),classes=classes,n_tasks=max(r['task'] for r in rows)+1,counts=dict(counts),task_counts={f'{k[0]}/{k[1]}':v for k,v in bytask.items()},manifest_sha256=sha(root/'manifest.jsonl'),source=source,protocol=protocol,seed=42,sample_decode_checks=len(examples))
 (root/'ready.json.partial').write_text(json.dumps(info,indent=2));(root/'ready.json.partial').replace(root/'ready.json');print(json.dumps(info|{'classes':len(classes)}),flush=True)

def imagenet(a):
 from datasets import Dataset,concatenate_datasets,Image as HFImage
 cache=Path(a.source).resolve();rows=[];classes=None;counts={}
 for official in ['train','validation']:
  files=sorted(cache.glob(f'imagenet-100-{official}*.arrow'));assert files,(cache,official)
  ds=concatenate_datasets([Dataset.from_file(str(f)) for f in files]).cast_column('image',HFImage(decode=False))
  classes=ds.features['label'].names;assert len(classes)==100;counts[official]=len(ds)
  # Labels are read without decoding or rewriting the compressed image bytes.
  for i,label in enumerate(ds['label']):
   label=int(label);split='test' if official=='validation' else 'validation' if bucket(f'imagenet/{official}/{i}/{label}')==0 else 'train'
   rows.append(dict(path=str(cache)+f'#{official}/{i}',label=label,task=0,split=split,storage='hf_arrow'))
  assert set(ds['label'])==set(range(100));print(json.dumps(dict(official_split=official,examples=len(ds),classes=100)),flush=True)
 assert counts=={'train':126689,'validation':5000},counts
 publish(Path(a.root)/'imagenet100',rows,classes,dict(original_cache=str(cache),dataset='https://huggingface.co/datasets/clane9/imagenet-100',official_counts=counts,dataset_info_sha256=sha(cache/'dataset_info.json')),'Fixed hash 10% of official train for validation; official validation used only as test. Direct Arrow access preserves the mirror shorter-side-160px pixels.',131689)

def domainnet(a):
 root=Path(a.root)/'domainnet';rows=[];archives={}
 for task,d in enumerate(DOMAINS):
  suffix=('groundtruth/' if d in ['clipart','painting'] else '')+d+'.zip';base='csr.bu.edu/ftp/visda/2019/multi-source/'
  archive=download([f'https://{base}{suffix}',f'http://{base}{suffix}'],root/'archives'/f'{d}.zip');extract_zip(archive,root/'raw');archives[d]=sha(archive)
  for official in ['train','test']:
   name=f'{d}_{official}.txt';f=download([f'https://{base}domainnet/txt/{name}',f'http://{base}domainnet/txt/{name}'],root/'splits'/name)
   for line in f.read_text().splitlines():
    path,label=line.rsplit(maxsplit=1);split='test' if official=='test' else 'validation' if bucket(path)==0 else 'train'
    rows.append(dict(path=str((root/'raw'/path).resolve()),label=int(label),task=task,split=split))
 assert len({r['label'] for r in rows})==345
 publish(root,rows,[str(i) for i in range(345)],dict(page='https://ai.bu.edu/M3SDA/',archive_sha256=archives),'Cleaned official six domains in fixed order clipart,infograph,painting,quickdraw,real,sketch; hash 10% of each official train for validation; official test untouched.',586575)

def core50(a):
 root=Path(a.root)/'core50';archive=download(['https://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip','http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'],root/'archives/core50_128x128.zip');extract_zip(archive,root/'raw')
 lists=download(['https://vlomonaco.github.io/core50/data/batches_filelists.zip'],root/'archives/batches_filelists.zip');extract_zip(lists,root/'splits')
 directories=[f.parent for f in (root/'splits').rglob('train_batch_00_filelist.txt') if 'nc_inc' in str(f).lower() and 'run0' in str(f).lower()]
 assert len(directories)==1,directories;folder=directories[0];rows=[];label_task={}
 def resolve(s):
  for path in [root/'raw'/s,root/'raw/core50_128x128'/s]:
   if path.exists():return str(path.resolve())
  raise FileNotFoundError(s)
 for task,f in enumerate(sorted(folder.glob('train_batch_*_filelist.txt'))):
  for line in f.read_text().splitlines():
   path,label=line.rsplit(maxsplit=1);label_task.setdefault(int(label),task);assert label_task[int(label)]==task;stem=Path(path).stem;frame=int(stem.split('_')[-1]);group=str(Path(path).parent)+f'/clip{frame//30}'
   rows.append(dict(path=resolve(path),label=int(label),task=task,split='validation' if bucket(group)==0 else 'train'))
 for line in (folder/'test_filelist.txt').read_text().splitlines():
  path,label=line.rsplit(maxsplit=1);rows.append(dict(path=resolve(path),label=int(label),task=label_task[int(label)],split='test'))
 assert len({r['label'] for r in rows})==50
 publish(root,rows,[f'object_{i}' for i in range(50)],dict(page='https://vlomonaco.github.io/core50/',image_archive_sha256=sha(archive),filelists_sha256=sha(lists)),'Official NC_inc Run0 class-incremental batches and fixed test sessions; seed42. Hash-selected 30-frame clip groups from training batches form validation; official test unchanged.',164866)

def tiny(a):
 src=Path(a.source);classes=(src/'wnids.txt').read_text().splitlines();assert len(classes)==200;idx={s:i for i,s in enumerate(classes)};rows=[]
 for cl in classes:
  for f in sorted((src/'train'/cl/'images').glob('*')):
   rows.append(dict(path=str(f.resolve()),label=idx[cl],task=0,split='validation' if bucket(f.name)==0 else 'train'))
 for line in (src/'val/val_annotations.txt').read_text().splitlines():
  image,cl,*_=line.split();rows.append(dict(path=str((src/'val/images'/image).resolve()),label=idx[cl],task=0,split='test'))
 publish(Path(a.root)/'tinyimagenet',rows,classes,dict(existing_root=str(src)),'Hash 10% of official train for validation; official labeled validation reserved for test.',110000)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['imagenet100','domainnet','core50','tinyimagenet']);p.add_argument('--root',required=True);p.add_argument('--source');a=p.parse_args();{'imagenet100':imagenet,'domainnet':domainnet,'core50':core50,'tinyimagenet':tiny}[a.dataset](a)
