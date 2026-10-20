"""Leakage-aware preprocessing, training-only TabDDPM-style augmentation, and FT-Transformer training."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_ft_transformer import FTTransformer

NUMERICAL = ["AGEATVIS", "HEIGHT", "WEIGHT", "BMI", "TOTCHOL", "HDL", "TRIGLYC",
 "NONHDL", "MHLDLC", "ALBU", "PROT", "GLOBULIN", "A_G_RATIO", "TC_HDL_RATIO",
 "LDL_HDL_RATIO", "TG_HDL_RATIO", "HBA1C", "GLUCOSE", "CREAT", "AST", "ALT",
 "ART_DURATION_DAYS", "CD4N", "CD4P", "CD8N", "CD8P", "HSRAT"]
CATEGORICAL = ["SITE", "STUDYHIST", "VISTYPE", "SEXBIRTH", "FASTING", "CHOLRX",
 "NIACIN", "FIBRATES", "DIABETES", "ONART", "ARTV", "HAARTV2", "ADH"]

# Broad physiologic/unit-consistency bounds. Values are set missing only when they are
# both beyond a 3-IQR training fence and outside these plausible ranges.
CLINICAL_LIMITS = {
 "AGEATVIS": (18, 100), "HEIGHT": (1.2, 2.2), "WEIGHT": (30, 250), "BMI": (10, 80),
 "TOTCHOL": (50, 500), "HDL": (5, 200), "TRIGLYC": (10, 2000), "NONHDL": (10, 450),
 "MHLDLC": (5, 400), "ALBU": (1, 7), "PROT": (2, 12), "GLOBULIN": (.5, 8),
 "A_G_RATIO": (.1, 5), "TC_HDL_RATIO": (.5, 20), "LDL_HDL_RATIO": (.1, 15),
 "TG_HDL_RATIO": (.1, 50), "HBA1C": (2, 25), "GLUCOSE": (30, 700),
 "CREAT": (.1, 20), "AST": (1, 2000), "ALT": (1, 2000), "ART_DURATION_DAYS": (0, 15000),
 "CD4N": (0, 5000), "CD4P": (0, 100), "CD8N": (0, 10000), "CD8P": (0, 100), "HSRAT": (0, 10),
}

class StructuredPreprocessor:
    def __init__(self, missing_threshold=.50, seed=42):
        self.missing_threshold, self.seed = missing_threshold, seed

    def fit(self, train):
        self.num_cols = [c for c in NUMERICAL if train[c].isna().mean() <= self.missing_threshold]
        self.cat_cols = [c for c in CATEGORICAL if train[c].isna().mean() <= self.missing_threshold]
        self.excluded_missing = [c for c in NUMERICAL + CATEGORICAL if c not in self.num_cols + self.cat_cols]
        x = train[self.num_cols].apply(pd.to_numeric, errors="coerce").copy()
        self.iqr_fences, self.outlier_counts = {}, {}
        for c in self.num_cols:
            q1, q3 = x[c].quantile([.25, .75]); iqr = q3-q1
            self.iqr_fences[c] = [float(q1-3*iqr), float(q3+3*iqr)]
            lo, hi = CLINICAL_LIMITS[c]; flo, fhi = self.iqr_fences[c]
            bad = ((x[c] < flo) & (x[c] < lo)) | ((x[c] > fhi) & (x[c] > hi))
            self.outlier_counts[c] = int(bad.sum()); x.loc[bad, c] = np.nan
        skew = x.skew()
        self.log_cols = [c for c in self.num_cols if skew[c] > 1 and x[c].min(skipna=True) >= 0]
        x.loc[:, self.log_cols] = np.log1p(x[self.log_cols])
        self.imputer = IterativeImputer(random_state=self.seed, max_iter=15, initial_strategy="median",
                                        skip_complete=True).fit(x)
        xi = self.imputer.transform(x)
        self.means = xi.mean(0); self.stds = xi.std(0); self.stds[self.stds == 0] = 1
        self.modes, self.maps = {}, {}
        for c in self.cat_cols:
            mode = train[c].dropna().astype(str).mode()
            self.modes[c] = mode.iloc[0] if len(mode) else "MISSING"
            vals = sorted(train[c].dropna().astype(str).unique().tolist())
            if self.modes[c] not in vals: vals.append(self.modes[c])
            self.maps[c] = {v:i for i,v in enumerate(vals)}
        return self

    def _clean_num(self, df):
        x = df[self.num_cols].apply(pd.to_numeric, errors="coerce").copy()
        for c in self.num_cols:
            flo, fhi = self.iqr_fences[c]; lo, hi = CLINICAL_LIMITS[c]
            bad = ((x[c] < flo) & (x[c] < lo)) | ((x[c] > fhi) & (x[c] > hi))
            x.loc[bad, c] = np.nan
        x.loc[:, self.log_cols] = np.log1p(x[self.log_cols].clip(lower=0))
        return x

    def transform(self, df):
        xi = self.imputer.transform(self._clean_num(df))
        xn = ((xi-self.means)/self.stds).astype(np.float32)
        xc = np.stack([df[c].astype("string").fillna(self.modes[c]).astype(str)
                       .map(self.maps[c]).fillna(self.maps[c][self.modes[c]]).to_numpy(np.int64)
                       for c in self.cat_cols], 1)
        return xn, xc

    @property
    def cardinalities(self): return [len(self.maps[c]) for c in self.cat_cols]
    def report(self, train):
        return {"missing_threshold":self.missing_threshold,
          "training_missing_fraction":{c:float(train[c].isna().mean()) for c in NUMERICAL+CATEGORICAL},
          "excluded_for_missingness":self.excluded_missing, "log1p_columns":self.log_cols,
          "training_outliers_set_missing":self.outlier_counts, "numerical_features":self.num_cols,
          "categorical_features":self.cat_cols,
          "note":"VLOAD and viral-load flags are excluded because they define/proxy the ART outcome."}
    def state(self):
        return {"num_cols":self.num_cols,"cat_cols":self.cat_cols,"iqr_fences":self.iqr_fences,
          "clinical_limits":CLINICAL_LIMITS,"log_cols":self.log_cols,"imputer":self.imputer,
          "means":self.means,"stds":self.stds,"modes":self.modes,"maps":self.maps}

class Denoiser(nn.Module):
    def __init__(self, dim, steps=100):
        super().__init__(); self.t = nn.Embedding(steps, 32)
        self.net = nn.Sequential(nn.Linear(dim+32,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,dim))
    def forward(self,x,t): return self.net(torch.cat([x,self.t(t)],1))

def tabddpm_augment(xn, xc, n_new, seed=42, epochs=250, steps=100):
    """TabDDPM-style Gaussian diffusion over continuous + one-hot categorical training data."""
    rng=np.random.default_rng(seed); cards=[int(xc[:,i].max()+1) for i in range(xc.shape[1])]
    onehot=np.concatenate([np.eye(k,dtype=np.float32)[xc[:,i]] for i,k in enumerate(cards)],1)
    raw=np.concatenate([xn,onehot],1).astype(np.float32); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=torch.tensor(raw,device=device); model=Denoiser(raw.shape[1],steps).to(device)
    beta=torch.linspace(1e-4,.02,steps,device=device); alpha=1-beta; abar=torch.cumprod(alpha,0)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5); gen=torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(epochs):
        idx=torch.randint(len(data),(min(256,len(data)),),generator=gen,device=device); x0=data[idx]
        t=torch.randint(steps,(len(idx),),generator=gen,device=device); noise=torch.randn(x0.shape,generator=gen,device=device)
        xt=abar[t,None].sqrt()*x0+(1-abar[t,None]).sqrt()*noise
        loss=nn.functional.mse_loss(model(xt,t),noise); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); x=torch.randn((n_new,raw.shape[1]),generator=gen,device=device)
    with torch.no_grad():
        for ti in reversed(range(steps)):
            t=torch.full((n_new,),ti,device=device,dtype=torch.long); eps=model(x,t)
            x=(x-(1-alpha[ti])/torch.sqrt(1-abar[ti])*eps)/torch.sqrt(alpha[ti])
            if ti: x += torch.sqrt(beta[ti])*torch.randn(x.shape,generator=gen,device=device)
    z=x.cpu().numpy(); syn_n=np.clip(z[:,:xn.shape[1]],-5,5).astype(np.float32); pos=xn.shape[1]; cats=[]
    for k in cards: cats.append(np.argmax(z[:,pos:pos+k],1)); pos+=k
    return syn_n,np.stack(cats,1).astype(np.int64)

def make_dataset(xn,xc,yh,ya,mask):
    return TensorDataset(torch.tensor(xn),torch.tensor(xc),torch.tensor(yh,dtype=torch.float32),
      torch.tensor(ya,dtype=torch.float32),torch.tensor(mask,dtype=torch.float32))
def loss_fn(hl,al,yh,ya,m):
    h=nn.functional.binary_cross_entropy_with_logits(hl,yh); r=nn.functional.binary_cross_entropy_with_logits(al,ya,reduction="none")
    a=(r*m).sum()/m.sum().clamp_min(1); return h+a
@torch.no_grad()
def predict(model,loader,device):
    model.eval(); out=[]
    for b in loader:
        xn,xc,yh,ya,m=[z.to(device) for z in b]; hl,al=model(xn,xc)
        out.append((torch.sigmoid(hl).cpu(),torch.sigmoid(al).cpu(),yh.cpu(),ya.cpu(),m.cpu()))
    return [torch.cat(z).numpy() for z in zip(*out)]
def score(y,p):
    q=(p>=.5).astype(int); return {"n":len(y),"prevalence":float(y.mean()),"roc_auc":roc_auc_score(y,p),
      "average_precision":average_precision_score(y,p),"accuracy":accuracy_score(y,q),
      "balanced_accuracy":balanced_accuracy_score(y,q),"f1":f1_score(y,q,zero_division=0),
      "brier":brier_score_loss(y,p),"confusion_matrix":confusion_matrix(y,q,labels=[0,1]).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="datasets/MWCCS_full_5844.csv")
    ap.add_argument("--output",default="artifacts_preprocessed"); ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--patience",type=int,default=10); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--ddpm-epochs",type=int,default=250); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.output); out.mkdir(exist_ok=True); df=pd.read_csv(args.data).drop_duplicates().copy()
    parts={s:df[df.DATA_SPLIT.eq(s)].reset_index(drop=True) for s in ["train","validation","test"]}; prep=StructuredPreprocessor(seed=args.seed).fit(parts["train"])
    (out/"preprocessing_report.json").write_text(json.dumps(prep.report(parts["train"]),indent=2)); joblib.dump(prep.state(),out/"preprocessor.joblib")
    arrays={}
    for s,d in parts.items():
        xn,xc=prep.transform(d); arrays[s]=[xn,xc,d.HIV_SCREEN_LABEL.to_numpy(np.float32),
          d.ART_UNSUPPRESSED_LABEL.fillna(0).to_numpy(np.float32),d.ART_MONITORING_ELIGIBLE.to_numpy(np.float32)]
    xn,xc,yh,ya,m=arrays["train"]; eligible=m.astype(bool); minority=eligible&(ya==1); majority=eligible&(ya==0)
    ratio=int(majority.sum())/max(int(minority.sum()),1); n_new=max(0,int(majority.sum()/2-minority.sum())) if ratio>=4 else 0
    if n_new:
        sn,sc=tabddpm_augment(xn[minority],xc[minority],n_new,args.seed,args.ddpm_epochs)
        arrays["train"]=[np.r_[xn,sn],np.r_[xc,sc],np.r_[yh,np.ones(n_new,dtype=np.float32)],
          np.r_[ya,np.ones(n_new,dtype=np.float32)],np.r_[m,np.ones(n_new,dtype=np.float32)]]
    aug={"method":"training-only TabDDPM-style Gaussian diffusion","imbalance_trigger_ratio":4,
      "eligible_majority":int(majority.sum()),"eligible_minority":int(minority.sum()),"synthetic_minority_rows":n_new,
      "post_augmentation_minority_to_majority":float((minority.sum()+n_new)/majority.sum())}
    (out/"augmentation_report.json").write_text(json.dumps(aug,indent=2))
    loaders={s:DataLoader(make_dataset(*a),batch_size=64,shuffle=s=="train") for s,a in arrays.items()}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=FTTransformer(len(prep.num_cols),prep.cardinalities).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5); best=float("inf"); stale=0; hist=[]
    for ep in range(1,args.epochs+1):
        model.train(); sm=nn_=0
        for b in loaders["train"]:
            z=[v.to(device) for v in b]; opt.zero_grad(); loss=loss_fn(*model(z[0],z[1]),*z[2:]); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); sm+=loss.item()*len(z[0]); nn_+=len(z[0])
        model.eval(); vs=vn=0
        with torch.no_grad():
            for b in loaders["validation"]:
                z=[v.to(device) for v in b]; loss=loss_fn(*model(z[0],z[1]),*z[2:]); vs+=loss.item()*len(z[0]); vn+=len(z[0])
        rec={"epoch":ep,"train_loss":sm/nn_,"validation_loss":vs/vn}; hist.append(rec); print(json.dumps(rec),flush=True)
        if rec["validation_loss"]<best-1e-5:
            best=rec["validation_loss"]; stale=0; torch.save({"model_state":model.state_dict(),"cardinalities":prep.cardinalities,
              "numerical":prep.num_cols,"categorical":prep.cat_cols,"config":vars(args)},out/"best_model.pt")
        else:
            stale+=1
            if stale>=args.patience: break
    c=torch.load(out/"best_model.pt",map_location=device,weights_only=False); model.load_state_dict(c["model_state"])
    ph,pa,yh,ya,m=predict(model,loaders["test"],device); e=m.astype(bool)
    result={"device":str(device),"best_epoch":int(np.argmin([r["validation_loss"] for r in hist])+1),"best_validation_loss":best,
      "hiv_screening":score(yh.astype(int),ph),"art_monitoring_eligible_only":score(ya[e].astype(int),pa[e]),"augmentation":aug}
    pd.DataFrame(hist).to_csv(out/"training_history.csv",index=False); (out/"metrics.json").write_text(json.dumps(result,indent=2))
    pred=parts["test"][["CCSID"]].copy(); pred["hiv_label"]=yh.astype(int); pred["hiv_probability"]=ph
    pred["art_eligible"]=m.astype(int); pred["art_label"]=np.where(e,ya,np.nan); pred["art_probability"]=pa; pred.to_csv(out/"test_predictions.csv",index=False)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
