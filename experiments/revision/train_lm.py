"""From-scratch GPT2 LM; 1000 predefined evaluation units, matched token budget.
WikiText103 uses the original training machine's copied Arrow cache. Held-out
splits never enter tokenization fitting or training. No pretrained weights.
"""
import argparse,fcntl,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import GPT2Config,GPT2LMHeadModel,GPT2TokenizerFast,GPTNeoConfig,GPTNeoForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stream import optimizer,flag_snapshot

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['gpt2-small','gpt2-medium','decoder125m'],default='gpt2-small'); p.add_argument('--dataset',choices=['wikitext103','openwebtext','c4'],default='wikitext103'); p.add_argument('--method',required=True); p.add_argument('--rank',type=int,default=64); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--units',type=int,default=1000); p.add_argument('--steps-per-unit',type=int,default=10); p.add_argument('--batch',type=int,default=8); p.add_argument('--length',type=int,default=256); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args(); assert a.units>=1000 or a.smoke
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42); root=Path(a.out); root.mkdir(parents=True,exist_ok=True); (root/'checkpoints').mkdir(exist_ok=True); (root/'flags').mkdir(exist_ok=True)
    if (root/'summary.json').exists(): return
    tokpath=Path(a.data)/'tokenizer_gpt2'
    tok=GPT2TokenizerFast.from_pretrained(str(tokpath) if tokpath.exists() else 'gpt2',local_files_only=tokpath.exists()); cache=Path(a.data)/f'tokens_{a.dataset}_gpt2.pt'
    with (Path(a.data)/'tokens_prepare.lock').open('a+') as token_lock:
        fcntl.flock(token_lock,fcntl.LOCK_EX)
        if cache.exists(): data=torch.load(cache,weights_only=False)
        else:
            portable={s:Path(a.data)/a.dataset/f'{s}.jsonl' for s in ['train','validation','test']}
            if all(f.exists() for f in portable.values()):
                text={s:(json.loads(line)['text'] for line in f.open()) for s,f in portable.items()}
            elif a.dataset=='wikitext103':
                loc=next((Path(a.data)/'hf_datasets'/'Salesforce___wikitext'/'wikitext-103-raw-v1'/'0.0.0').iterdir())
                splits={s: [loc/f'wikitext-{s}.arrow'] for s in ['validation','test']}; splits['train']=sorted(loc.glob('wikitext-train*.arrow'))
                text={s:(str(t) for f in files for t in Dataset.from_file(str(f))['text']) for s,files in splits.items()}
            else:
                # Frozen corpus subsets must be materialized before training. The manifest
                # records their provenance and split hashes; do not invent held-out splits.
                files={s:Path(a.data)/a.dataset/f'{s}.jsonl' for s in ['train','validation','test']}
                if not all(f.exists() for f in files.values()): raise FileNotFoundError(f'Missing materialized {a.dataset} split files: {files}')
                text={s:(json.loads(line)['text'] for line in f.open()) for s,f in files.items()}
            data={}
            for s,lines in text.items():
                ids=[]; batch=[]
                for line in lines:
                    batch.append(line)
                    if len(batch)>=1000:
                        for seq in tok(batch,add_special_tokens=False,verbose=False)['input_ids']: ids.extend(seq+[tok.eos_token_id])
                        batch=[]
                if batch:
                    for seq in tok(batch,add_special_tokens=False,verbose=False)['input_ids']: ids.extend(seq+[tok.eos_token_id])
                data[s]=torch.tensor(ids,dtype=torch.int32)
            torch.save(data,cache)
    cfg=GPT2Config(vocab_size=50257,n_positions=a.length,n_ctx=a.length,n_embd=1024 if a.model=='gpt2-medium' else 768,n_layer=24 if a.model=='gpt2-medium' else 12,n_head=16 if a.model=='gpt2-medium' else 12,resid_pdrop=.1,embd_pdrop=.1,attn_pdrop=.1,use_cache=False)
    if a.smoke: cfg.n_embd=128; cfg.n_layer=2; cfg.n_head=4
    if a.model=='decoder125m':
        cfg=GPTNeoConfig(vocab_size=50257,max_position_embeddings=a.length,hidden_size=128 if a.smoke else 768,num_layers=2 if a.smoke else 12,num_heads=4 if a.smoke else 12,intermediate_size=512 if a.smoke else 3072,attention_types=[[['global','local'],1 if a.smoke else 6]],window_size=min(256,a.length))
        model=GPTNeoForCausalLM(cfg).cuda()
    else: model=GPT2LMHeadModel(cfg).cuda()
    opt=optimizer(model,a); total=a.units*a.steps_per_unit; step=0; best=float('inf'); prev={}; start=0
    info=vars(a)|dict(seed=42,parameters=sum(p.numel() for p in model.parameters()),token_budget=total*a.batch*a.length,split_tokens={s:len(v) for s,v in data.items()},epoch_equivalent_definition='one unit = fixed 10 optimizer steps, NOT a corpus epoch',precision='float32',selection='minimum validation loss')
    (root/'config.json').write_text(json.dumps(info,indent=2))
    resume=root/'checkpoints'/'latest.pt'
    if resume.exists():
        ck=torch.load(resume,map_location='cuda',weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start=ck['unit']; step=ck['step']; best=ck['best']
    if start==0: torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),unit=0,step=0),root/'checkpoints'/'unit_0000.pt')
    @torch.no_grad()
    def evaluate(split):
        model.eval(); x=data[split]; losses=[]
        # Complete held-out splits, evaluated in fixed contiguous blocks.
        for offset in range(0,len(x)-a.length-1,a.batch*a.length):
            n=min(a.batch,(len(x)-offset-1)//a.length)
            if n<=0: break
            xx=x[offset:offset+n*a.length].long().view(n,a.length).cuda(); yy=x[offset+1:offset+1+n*a.length].long().view(n,a.length).cuda()
            logits=model(xx).logits; losses.append((float(torch.nn.functional.cross_entropy(logits.reshape(-1,50257),yy.reshape(-1))),n*a.length))
            if a.smoke: break
        return sum(l*n for l,n in losses)/sum(n for _,n in losses)
    final={}
    for unit in range(start+1,a.units+1):
        t0=time.monotonic(); model.train(); train_loss=gn=0.; logs=[]; torch.manual_seed(42+unit); torch.cuda.reset_peak_memory_stats()
        for _ in range(a.steps_per_unit):
            starts=torch.randint(len(data['train'])-a.length-1,(a.batch,)); xx=torch.stack([data['train'][i:i+a.length] for i in starts]).long().cuda(); yy=torch.stack([data['train'][i+1:i+a.length+1] for i in starts]).long().cuda()
            opt.zero_grad(set_to_none=True); logits=model(xx).logits; loss=torch.nn.functional.cross_entropy(logits.reshape(-1,50257),yy.reshape(-1)); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            lr=a.lr*min(1.,(step+1)/100)*.5*(1+math.cos(math.pi*step/total))
            for g in opt.param_groups: g['lr']=lr
            opt.step(); step+=1; train_loss+=float(loss); gn+=float(grad)
            if any(s.get('t',-1)%50==1 for s in opt.state.values() if 'U' in s): logs.append(flag_snapshot(opt,root/'flags',step,prev))
        va=evaluate('validation'); te=evaluate('test'); final=dict(unit=unit,tokens=step*a.batch*a.length,train_loss=train_loss/a.steps_per_unit,val_loss=va,val_ppl=math.exp(min(va,50)),test_loss=te,test_ppl=math.exp(min(te,50)),gradient_norm=gn/a.steps_per_unit,lr=lr,wall_seconds=time.monotonic()-t0,peak_bytes=torch.cuda.max_memory_allocated(),flag_diagnostics=logs)
        with (root/'log.jsonl').open('a') as f:f.write(json.dumps(final)+'\n')
        if va<best: best=va; torch.save(dict(model=model.state_dict(),unit=unit,val_loss=va),root/'checkpoints'/'best_val.pt')
        if unit%10==0 or unit in [1,2,5,20,50,100,200,500,1000] or unit==a.units:
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),unit=unit,step=step,best=best); torch.save(ck,root/'checkpoints'/f'unit_{unit:04d}.pt'); torch.save(ck,resume)
        print(json.dumps({k:final[k] for k in ['unit','val_ppl','wall_seconds']}),flush=True)
    (root/'summary.json').write_text(json.dumps(dict(complete=True,best_val_ppl=math.exp(min(best,50)),final=final),indent=2))
if __name__=='__main__':main()
