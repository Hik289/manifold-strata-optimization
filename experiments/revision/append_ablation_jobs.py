import fcntl,json
from pathlib import Path
q=Path("revision/queue.json")
with q.with_suffix(".lock").open("a+") as f:
 fcntl.flock(f,fcntl.LOCK_EX);jobs=json.loads(q.read_text());outs={j["out"] for j in jobs};nid=max(j["id"] for j in jobs)+1
 for ab in ["same_rho","reverse_rho","equal_a","no_residual","random_flag","no_transport","frozen_after_warmup"]:
  for lr in [.0003,.001]:
   out=f"runs_revision/rotated_mnist_cl__finetune__strata__{ab}__lr{lr}"
   if out in outs:continue
   jobs.append(dict(id=nid,family="ablation",out=out,args=["revision/train_stream.py","--task","rotated_mnist_cl","--method","strata","--ablation",ab,"--lr",str(lr),"--epochs","1000","--data","../data","--out",out],status="pending"));nid+=1
 q.write_text(json.dumps(jobs,indent=2));print(len(jobs))
