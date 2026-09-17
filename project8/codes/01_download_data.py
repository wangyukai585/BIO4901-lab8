#!/usr/bin/env python
"""Download official FASTA, preserve official split, export whole-data QC."""
import hashlib, time, urllib.request
from utils import *
URL='https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output_dir',type=Path,default=ROOT/'results/qc'); p.add_argument('--data',type=Path,default=ROOT/'data/raw/deeploc_data.fasta'); a=p.parse_args()
    a.data.parent.mkdir(parents=True,exist_ok=True); a.output_dir.mkdir(parents=True,exist_ok=True)
    if not a.data.exists():
        temp=a.data.with_suffix('.partial')
        for attempt in range(3):
            try:
                print(f'Downloading {URL}, attempt {attempt+1}',flush=True)
                with urllib.request.urlopen(URL,timeout=60) as response,open(temp,'wb') as f:
                    while chunk:=response.read(1024*1024): f.write(chunk); print(f'{f.tell()/2**20:.1f} MiB',flush=True)
                parse_fasta(temp); temp.replace(a.data); break
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    df=parse_fasta(a.data); audit=leakage_check(df)
    write_json(a.output_dir/'leakage_audit.json',audit)
    counts=df.groupby(['split','location']).size().unstack(0).reindex(CLASSES)
    counts.to_csv(a.output_dir/'class_distribution.csv')
    lengths=df.groupby('split').length.describe(percentiles=[.25,.5,.75,.9,.95,.99])
    lengths.to_csv(a.output_dir/'length_distribution.csv')
    write_json(a.output_dir/'summary.json',{'source':URL,'sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'n':len(df),'raw_n':sum(1 for _ in SeqIO.parse(a.data,'fasta')),'excluded_dual_localization':sum('Cytoplasm-Nucleus' in r.description for r in SeqIO.parse(a.data,'fasta')),'label_mapping':{'Plastid':'Chloroplast (broader source annotation)'},'class_counts':df.location.value_counts().to_dict(),'imbalance_max_min':float(df.location.value_counts().max()/df.location.value_counts().min()),'truncation_fraction':{str(n):float((df.length>n).mean()) for n in [128,512,1022]},'noncanonical_residue_count':int(df.sequence.str.count('[^ACDEFGHIKLMNPQRSTVWY]').sum()),'split_policy':'official test token only','audit':audit})
    # Individual sequences and metadata stay in ignored processed directory.
    dest=ROOT/'data/processed'; dest.mkdir(parents=True,exist_ok=True); df.to_csv(dest/'metadata.csv',index=False)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks',palette='deep',font_scale=.9)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    counts.plot.barh(ax=axs[0],color=sns.color_palette('deep')[:2]); axs[0].set(xlabel='Proteins',ylabel='',title='A  Official split: class counts'); axs[0].invert_yaxis()
    for split,color in zip(['train','test'],sns.color_palette('deep')):
        axs[1].hist(df[df.split==split].length,bins=np.geomspace(df.length.min(),df.length.max(),40),alpha=.65,label=split,color=color)
    axs[1].set(xscale='log',xlabel='Sequence length (residues)',ylabel='Proteins',title='B  Full dataset length distribution'); axs[1].legend(); sns.despine(); fig.tight_layout()
    image_dir=ROOT/'results/images'; image_dir.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']: fig.savefig(image_dir/f'fig0_data_qc.{ext}',dpi=300,bbox_inches='tight')
    print(json.dumps(audit,indent=2)); print(counts)
    if audit['cross_split_identical_sequence_hashes']: raise ValueError('Exact leakage detected')
if __name__=='__main__': main()
