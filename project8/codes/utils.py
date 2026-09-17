"""Shared reproducible data, device, logging and ESM utilities."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, logging, os, platform, random, re, sys, threading, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'facebook/esm2_t30_150M_UR50D'
# Immutable Hub revision; populated from the Hub once, also saved in run metadata.
REVISION = 'a695f6045e2e32885fa60af20c13cb35398ce30c'
CLASSES = ['Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell membrane',
           'Endoplasmic reticulum','Chloroplast','Golgi apparatus','Lysosome/Vacuole','Peroxisome']

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x: x.item() if hasattr(x,'item') else str(x)))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8,os.cpu_count() or 1))

def get_device(request='auto'):
    automatic=request=='auto'
    if automatic: request='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    d=torch.device(request)
    if d.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    if d.type=='mps' and not torch.backends.mps.is_available(): raise ValueError('MPS requested but unavailable')
    if d.type not in ('cuda','mps','cpu'): raise ValueError(f'Unsupported device {d}')
    try:
        x=torch.ones(2,device=d); _=x.sum().item()
    except RuntimeError:
        if not automatic: raise
        d=torch.device('cpu')
    return d

def sync(d):
    if d.type=='cuda': torch.cuda.synchronize(d)
    elif d.type=='mps': torch.mps.synchronize()

def memory(d):
    import psutil
    return {'gpu_allocated_mb':torch.cuda.memory_allocated(d)/2**20 if d.type=='cuda' else None,
            'gpu_peak_mb':torch.cuda.max_memory_allocated(d)/2**20 if d.type=='cuda' else None,
            'mps_allocated_mb':torch.mps.current_allocated_memory()/2**20 if d.type=='mps' else None,
            'rss_mb':psutil.Process().memory_info().rss/2**20}

def clear_device(d):
    import gc
    gc.collect()
    if d.type=='cuda': torch.cuda.empty_cache()
    elif d.type=='mps': torch.mps.empty_cache()

def logger(out):
    Path(out).mkdir(parents=True,exist_ok=True)
    log=logging.getLogger(str(Path(out).resolve())); log.setLevel(logging.INFO); log.propagate=False
    if not log.handlers:
        for h in (logging.StreamHandler(sys.stdout),logging.FileHandler(Path(out)/'training.log')):
            h.setFormatter(logging.Formatter('%(asctime)s %(message)s')); log.addHandler(h)
    return log

class Monitor:
    """Heartbeat continues during long batches/downloads; ETA becomes measured."""
    def __init__(self,log,d,total,estimate=2.):
        self.log,self.d,self.total=log,d,total; self.start=time.monotonic(); self.done=0
        self.state={'epoch':0,'loss':None,'accuracy':None}; self.stop_event=threading.Event()
        log.info('device=%s estimated_seconds=%.1f (rough initial estimate; updated after first batch)',d,total*estimate)
    def emit(self):
        elapsed=time.monotonic()-self.start
        self.log.info(json.dumps(dict(self.state,step=self.done,total_steps=self.total,elapsed_s=round(elapsed,2),eta_s=round(elapsed/self.done*(self.total-self.done),2) if self.done else None,**memory(self.d))))
    def update(self,done,**state): self.done=done; self.state.update(state); self.emit()
    def __enter__(self):
        def beat():
            while not self.stop_event.wait(30): self.emit()
        self.thread=threading.Thread(target=beat,daemon=True); self.thread.start(); return self
    def __exit__(self,*exc): self.stop_event.set(); self.thread.join(); self.emit()

def parse_fasta(path):
    rows=[]
    for rec in SeqIO.parse(path,'fasta'):
        match=re.fullmatch(r'(\S+)\s+(.+)-([SMU])(?:\s+(test))?',rec.description.strip())
        if not match: raise ValueError(f'Invalid header: {rec.description}')
        acc,loc,mem,test=match.groups(); loc=loc.replace('.',' ')
        if loc=='Cytoplasm-Nucleus': continue  # Ambiguous dual localization excluded from 10-way task.
        if loc=='Plastid': loc='Chloroplast'  # Course label; original annotation is broader.
        if loc not in CLASSES: raise ValueError(f'Unknown location: {loc}')
        seq=str(rec.seq).upper()
        if not seq or re.search('[^A-Z]',seq): raise ValueError(f'Invalid sequence {acc}')
        rows.append(dict(id=acc,sequence=seq,label=CLASSES.index(loc),location=loc,membrane=mem,split='test' if test else 'train',length=len(seq),sequence_hash=hashlib.sha256(seq.encode()).hexdigest()))
    df=pd.DataFrame(rows)
    if df.empty or df.id.duplicated().any(): raise ValueError('Empty FASTA or duplicate accessions')
    if set(df.split)!= {'train','test'}: raise ValueError('Both official splits required')
    return df

def leakage_check(df):
    tr=df[df.split=='train']; te=df[df.split=='test']
    overlap=sorted(set(tr.sequence_hash)&set(te.sequence_hash))
    return {'train_n':len(tr),'test_n':len(te),'cross_split_identical_sequence_hashes':len(overlap),
            'cross_split_identical_sequence_pairs':int(sum((tr.sequence_hash==h).sum()*(te.sequence_hash==h).sum() for h in overlap)),
            'cross_split_accession_overlap':len(set(tr.id)&set(te.id)),
            'within_train_duplicate_sequences':int(tr.sequence_hash.duplicated().sum()),
            'within_test_duplicate_sequences':int(te.sequence_hash.duplicated().sum()),
            'homology_evidence':'Original DeepLoc paper section 2.3.1: PSI-CD-HIT, 30% identity OR 1e-6 E-value; 80% shorter-sequence coverage; clusters assigned to five folds; one held out.',
            'limitation':'Exact duplicate audit only; does NOT independently establish a pairwise homology threshold or exclude ESM pretraining overlap.',
            'source':'https://doi.org/10.1093/bioinformatics/btx431'}

def cap_split(df,n,seed):
    if n is None or n>=len(df): return df.sort_values('id').reset_index(drop=True)
    if n<10: raise ValueError('subset/eval size must be >=10 to cover 10 classes')
    # Deterministic proportional sample WITHIN the existing split; never resplit.
    order=df.assign(key=df.id.map(lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())).sort_values('key')
    groups=[g for _,g in order.groupby('label',sort=True)]
    chosen=[g.index[0] for g in groups]
    rest=order.drop(chosen)
    # Remaining slots follow the native class distribution (fixed seed).
    chosen+=rest.sample(n-len(chosen),random_state=seed).index.tolist()
    return df.loc[chosen].sort_values('id').reset_index(drop=True)

def load_data(args):
    df=parse_fasta(args.data)
    audit=leakage_check(df)
    if audit['cross_split_identical_sequence_hashes'] or audit['cross_split_accession_overlap']:
        raise ValueError('Exact train/test overlap detected; inspect QC before training')
    return (cap_split(df[df.split=='train'],args.subset_size,args.seed),
            cap_split(df[df.split=='test'],args.eval_size if args.eval_size is not None else args.subset_size,args.seed))

def optional_int(v): return None if str(v).lower() in ('none','null','all') else int(v)

def parser(stage):
    p=argparse.ArgumentParser(description=stage)
    p.add_argument('--data',type=Path,default=ROOT/'data/raw/deeploc_data.fasta')
    p.add_argument('--subset_size',type=optional_int,default=None,help='Cap official TRAIN split; never changes split membership')
    p.add_argument('--eval_size',type=optional_int,default=None,help='Cap official TEST split; defaults to subset_size')
    p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--device',default='auto'); p.add_argument('--output_dir',type=Path,default=ROOT/'results'/stage)
    p.add_argument('--model',default=MODEL); p.add_argument('--revision',default=REVISION)
    p.add_argument('--max_length',type=int,default=512,help='Residues, excluding BOS/EOS; preserve both termini')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--log_steps',type=int,default=10)
    return p

def trim(seq,n):
    return seq if len(seq)<=n else seq[:(n+1)//2]+seq[-(n//2):]

def tokenize(tokenizer,seqs,max_length,d,perturb=False):
    seqs=[trim(s,max_length) for s in seqs]
    if perturb:
        seqs=['X'*min(10,len(s)//4)+s[min(10,len(s)//4):len(s)-min(10,len(s)//4)]+'X'*min(10,len(s)//4) for s in seqs]
    b=tokenizer(seqs,padding=True,return_tensors='pt',return_special_tokens_mask=True)
    return {k:v.to(d) for k,v in b.items()}

def pooled(backbone,batch):
    mask=batch['attention_mask'].bool() & ~batch['special_tokens_mask'].bool()
    h=backbone(**{k:v for k,v in batch.items() if k!='special_tokens_mask'}).last_hidden_state
    return (h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)

def load_backbone(args):
    tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    model=AutoModel.from_pretrained(args.model,revision=args.revision,add_pooling_layer=False)
    # ESM contact head is unused in sequence classification.
    if hasattr(model,'contact_head'):
        for p in model.contact_head.parameters(): p.requires_grad=False
    return tok,model

def precision(d,requested='auto'):
    if requested=='fp32': return None
    if d.type=='cuda' and torch.cuda.is_bf16_supported(): return torch.bfloat16
    if d.type=='mps':
        # Runtime detection: older macOS/PyTorch may reject or disable bf16.
        import warnings
        try:
            with warnings.catch_warnings(record=True) as notices:
                x=torch.ones((2,2),device=d,requires_grad=True)
                with torch.autocast('mps',dtype=torch.bfloat16): y=x@x
                y.sum().backward()
                supported=y.dtype==torch.bfloat16 and not notices
            del x,y
            if supported: return torch.bfloat16
        except (RuntimeError,TypeError): pass
    return None

def amp(d,dtype): return torch.autocast(device_type=d.type,dtype=dtype) if dtype else contextlib.nullcontext()

def save_predictions(path,df,probs):
    if probs.shape!=(len(df),10) or not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1,atol=1e-4): raise ValueError('Invalid probabilities')
    out=df[['id','label','length','membrane','split']].copy(); out['prediction']=probs.argmax(1)
    for k in range(10): out[f'p{k}']=probs[:,k]
    out.to_csv(path,index=False)

def run_metadata(args,d,**extra):
    import transformers, peft
    return dict(config=vars(args),device=str(d),platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,status='SMOKE_PLACEHOLDER' if args.subset_size else 'FULL_RUN',**memory(d),**extra)
