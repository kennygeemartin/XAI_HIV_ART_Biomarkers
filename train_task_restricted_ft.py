"""Final multitask FT-Transformer with leakage-safe HIV and ART-expanded token sequences."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import train_ft_transformer_preprocessed as pp

COMMON_NUM=["AGEATVIS","HEIGHT","WEIGHT","BMI","TOTCHOL","HDL","TRIGLYC","NONHDL","MHLDLC",
 "ALBU","PROT","GLOBULIN","A_G_RATIO","TC_HDL_RATIO","LDL_HDL_RATIO","TG_HDL_RATIO","HBA1C",
 "GLUCOSE","CREAT","AST","ALT"]
ART_NUM=["ART_DURATION_DAYS","CD4N","CD4P","CD8N","CD8P","HSRAT"]
COMMON_CAT=["SEXBIRTH","FASTING","CHOLRX","NIACIN","FIBRATES","DIABETES"]
ART_CAT=["ONART","ARTV","HAARTV2","ADH"]
ALL_NUM=COMMON_NUM+ART_NUM; ALL_CAT=COMMON_CAT+ART_CAT

class TaskRestrictedFT(nn.Module):
    def __init__(self,cards,d=64,heads=8,blocks=3,ffn=128,dropout=.2):
        super().__init__(); self.n_common_num=len(COMMON_NUM); self.n_common_cat=len(COMMON_CAT)
        self.nw=nn.Parameter(torch.empty(len(ALL_NUM),d)); self.nb=nn.Parameter(torch.zeros(len(ALL_NUM),d)); nn.init.xavier_uniform_(self.nw)
        self.ce=nn.ModuleList([nn.Embedding(k,d) for k in cards]); self.cls=nn.Parameter(torch.zeros(1,1,d))
        layer=nn.TransformerEncoderLayer(d,heads,ffn,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,blocks,norm=nn.LayerNorm(d)); self.hiv_head=nn.Linear(d,1); self.art_head=nn.Linear(d,1)
    def forward(self,xn,xc,task):
        nn_used=self.n_common_num if task=='hiv' else len(ALL_NUM); nc_used=self.n_common_cat if task=='hiv' else len(ALL_CAT)
        nt=xn[:,:nn_used,None]*self.nw[:nn_used]+self.nb[:nn_used]
        ct=torch.stack([self.ce[i](xc[:,i]) for i in range(nc_used)],1); cls=self.cls.expand(len(xn),-1,-1)
        h=self.encoder(torch.cat([cls,nt,ct],1))[:,0]
        return (self.hiv_head(h) if task=='hiv' else self.art_head(h)).squeeze(1)

def ds(a): return TensorDataset(*[torch.tensor(x,dtype=torch.float32 if i!=1 else torch.long) for i,x in enumerate(a)])
def masked_loss(model,z):
    xn,xc,yh,ya,m=z; hl=model(xn,xc,'hiv'); al=model(xn,xc,'art')
    h=nn.functional.binary_cross_entropy_with_logits(hl,yh); r=nn.functional.binary_cross_entropy_with_logits(al,ya,reduction='none'); a=(r*m).sum()/m.sum().clamp_min(1)
    return h+a
@torch.no_grad()
def pred(model,loader,dev):
    model.eval(); out=[]
    for b in loader:
        z=[v.to(dev) for v in b]; out.append((torch.sigmoid(model(z[0],z[1],'hiv')).cpu(),torch.sigmoid(model(z[0],z[1],'art')).cpu(),z[2].cpu(),z[3].cpu(),z[4].cpu()))
    return [torch.cat(v).numpy() for v in zip(*out)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='datasets/MWCCS_full_5844.csv'); ap.add_argument('--output',default='artifacts_task_restricted')
    ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--patience',type=int,default=10); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--ddpm-epochs',type=int,default=250); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); out=Path(a.output); out.mkdir(exist_ok=True)
    df=pd.read_csv(a.data).drop_duplicates(); parts={s:df[df.DATA_SPLIT.eq(s)].reset_index(drop=True) for s in ['train','validation','test']}
    pp.NUMERICAL=ALL_NUM; pp.CATEGORICAL=ALL_CAT
    prep=pp.StructuredPreprocessor(seed=a.seed).fit(parts['train']); joblib.dump(prep.state(),out/'preprocessor.joblib')
    arrays={}
    for s,d in parts.items():
        xn,xc=prep.transform(d); arrays[s]=[xn,xc,d.HIV_SCREEN_LABEL.to_numpy(np.float32),d.ART_UNSUPPRESSED_LABEL.fillna(0).to_numpy(np.float32),d.ART_MONITORING_ELIGIBLE.to_numpy(np.float32)]
    xn,xc,yh,ya,m=arrays['train']; e=m.astype(bool); mi=e&(ya==1); ma=e&(ya==0); nnew=max(0,int(ma.sum()/2-mi.sum()))
    sn,sc=pp.tabddpm_augment(xn[mi],xc[mi],nnew,a.seed,a.ddpm_epochs)
    arrays['train']=[np.r_[xn,sn],np.r_[xc,sc],np.r_[yh,np.ones(nnew,np.float32)],np.r_[ya,np.ones(nnew,np.float32)],np.r_[m,np.ones(nnew,np.float32)]]
    loaders={s:DataLoader(ds(v),batch_size=64,shuffle=s=='train') for s,v in arrays.items()}; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TaskRestrictedFT(prep.cardinalities).to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5); best=float('inf'); stale=0; hist=[]
    for ep in range(1,a.epochs+1):
        model.train(); sm=n=0
        for b in loaders['train']:
            z=[v.to(dev) for v in b]; opt.zero_grad(); loss=masked_loss(model,z); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); sm+=loss.item()*len(z[0]); n+=len(z[0])
        model.eval(); vs=vn=0
        with torch.no_grad():
            for b in loaders['validation']:
                z=[v.to(dev) for v in b]; loss=masked_loss(model,z); vs+=loss.item()*len(z[0]); vn+=len(z[0])
        rec={'epoch':ep,'train_loss':sm/n,'validation_loss':vs/vn}; hist.append(rec); print(json.dumps(rec),flush=True)
        if rec['validation_loss']<best-1e-5:
            best=rec['validation_loss']; stale=0; torch.save({'model_state':model.state_dict(),'cardinalities':prep.cardinalities,'common_num':COMMON_NUM,'art_num':ART_NUM,'common_cat':COMMON_CAT,'art_cat':ART_CAT,'config':vars(a)},out/'best_model.pt')
        else:
            stale+=1
            if stale>=a.patience: break
    c=torch.load(out/'best_model.pt',map_location=dev,weights_only=False); model.load_state_dict(c['model_state']); ph,pa,yh,ya,m=pred(model,loaders['test'],dev); elig=m.astype(bool)
    from train_ft_transformer_preprocessed import score
    result={'device':str(dev),'best_epoch':int(np.argmin([x['validation_loss'] for x in hist])+1),'best_validation_loss':best,
      'hiv_screening_noninvasive':score(yh.astype(int),ph),'art_monitoring_eligible_only':score(ya[elig].astype(int),pa[elig]),
      'hiv_allowed_features':COMMON_NUM+COMMON_CAT,'art_additional_features':ART_NUM+ART_CAT,'synthetic_art_rows':nnew}
    pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False); (out/'metrics.json').write_text(json.dumps(result,indent=2))
    pr=parts['test'][['CCSID']].copy(); pr['hiv_label']=yh.astype(int); pr['hiv_probability']=ph; pr['art_eligible']=m.astype(int); pr['art_label']=np.where(elig,ya,np.nan); pr['art_probability']=pa; pr.to_csv(out/'test_predictions.csv',index=False)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
