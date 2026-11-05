"""Read-only bounded dataset inventory; prunes package and image-leaf trees."""
import argparse,json,os,re,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--seconds',type=int,default=100);p.add_argument('roots',nargs='+');a=p.parse_args();start=time.monotonic();hits=[];visited=0;errors=[]
pattern=re.compile(r'image[-_]?net|domain[-_]?net|co[-_]?re[-_]?50',re.I)
skip={'.git','node_modules','anaconda3','miniconda3','venv','.venv','site-packages','__pycache__','blobs','checkpoints','flags','proc','sys','dev','lost+found','Trash','.Trash','paper_db','review','reviews','old'}
for root in a.roots:
 if not os.path.isdir(root):continue
 for path,dirs,files in os.walk(root,onerror=lambda e:errors.append(str(e))):
  visited+=1;depth=len(Path(path).relative_to(root).parts)
  if time.monotonic()-start>a.seconds:dirs[:]=[];break
  dirs[:]=[d for d in dirs if d not in skip and not d.startswith('epoch_')]
  match=bool(pattern.search(Path(path).name))
  if match:
   hits.append(dict(path=path,kind='directory',entries=sorted(dirs+files)[:25]));print(json.dumps(hits[-1]),flush=True)
  for f in files:
   if pattern.search(f) and f.lower().endswith(('.zip','.tar','.gz','.tgz','.7z','.txt','.json','.csv','.parquet','.arrow','.yaml')):
    hits.append(dict(path=os.path.join(path,f),kind='file'));print(json.dumps(hits[-1]),flush=True)
  if depth>=7:dirs[:]=[]
  # No need to enumerate millions of individual class/image directories.
  if len(dirs)>40 and (sum(bool(re.fullmatch(r'n\d{8}',d)) for d in dirs)>40 or depth>=5):dirs[:]=[]
 print(json.dumps(dict(root_done=root,visited=visited,elapsed=time.monotonic()-start)),flush=True)
print(json.dumps(dict(host=a.host,candidates=hits,visited=visited,elapsed=time.monotonic()-start,errors=errors[:12],bounded=True)),flush=True)
