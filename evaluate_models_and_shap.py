"""Benchmark evaluation, publication charts, and permutation SHAP for the trained FT-Transformer."""
from __future__ import annotations
import json, random
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
 precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
from train_ft_transformer import FTTransformer
from train_ft_transformer_preprocessed import StructuredPreprocessor, tabddpm_augment

ROOT=Path("artifacts_preprocessed"); OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)
random.seed(42); np.random.seed(42); torch.manual_seed(42); sns.set_theme(style="whitegrid",context="paper")

def load_prep():
    s=joblib.load(ROOT/"preprocessor.joblib"); p=StructuredPreprocessor()
    for k,v in s.items(): setattr(p,k,v)
    return p
def met(y,p):
    q=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {"accuracy":accuracy_score(y,q),"sensitivity":recall_score(y,q,zero_division=0),
      "specificity":tn/(tn+fp),"precision":precision_score(y,q,zero_division=0),
      "f1":f1_score(y,q,zero_division=0),"auroc":roc_auc_score(y,p),
      "auprc":average_precision_score(y,p),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
def fit_benchmarks(Xtr,ytr):
    scale=float((ytr==0).sum()/max((ytr==1).sum(),1))
    models={
      "Logistic regression":LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42),
      "Random forest":RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight="balanced",n_jobs=-1,random_state=42),
      "XGBoost":XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,
        eval_metric="logloss",scale_pos_weight=scale,n_jobs=-1,random_state=42),
      "MLP":MLPClassifier(hidden_layer_sizes=(128,64),activation="relu",alpha=1e-5,batch_size=64,
        learning_rate_init=1e-3,max_iter=300,early_stopping=True,n_iter_no_change=10,random_state=42)}
    for m in models.values(): m.fit(Xtr,ytr)
    return models

df=pd.read_csv("datasets/MWCCS_full_5844.csv").drop_duplicates(); parts={s:df[df.DATA_SPLIT.eq(s)].reset_index(drop=True) for s in ["train","validation","test"]}
prep=load_prep(); arr={s:prep.transform(d) for s,d in parts.items()}
# One-hot categorical tokens for conventional models; fit categories from training only.
enc=OneHotEncoder(categories=[np.arange(k) for k in prep.cardinalities],handle_unknown="ignore",sparse_output=False)
enc.fit(arr["train"][1]); design={s:np.c_[x[0],enc.transform(x[1])].astype(np.float32) for s,x in arr.items()}

ck=torch.load(ROOT/"best_model.pt",map_location="cpu",weights_only=False); ft=FTTransformer(len(prep.num_cols),prep.cardinalities)
ft.load_state_dict(ck["model_state"]); ft.eval()
@torch.no_grad()
def ft_probs(xn,xc,task):
    h,a=ft(torch.tensor(xn,dtype=torch.float32),torch.tensor(xc,dtype=torch.long)); return torch.sigmoid(h if task=="HIV" else a).numpy()

rows=[]; probabilities={"HIV":{},"ART":{}}
for task in ["HIV","ART"]:
    if task=="HIV":
        trmask=np.ones(len(parts["train"]),bool); temask=np.ones(len(parts["test"]),bool)
        ytr=parts["train"].HIV_SCREEN_LABEL.to_numpy(int); yte=parts["test"].HIV_SCREEN_LABEL.to_numpy(int)
        Xtr=design["train"]; Xte=design["test"]
    else:
        trmask=parts["train"].ART_MONITORING_ELIGIBLE.eq(1).to_numpy(); temask=parts["test"].ART_MONITORING_ELIGIBLE.eq(1).to_numpy()
        ytr=parts["train"].loc[trmask,"ART_UNSUPPRESSED_LABEL"].to_numpy(int); yte=parts["test"].loc[temask,"ART_UNSUPPRESSED_LABEL"].to_numpy(int)
        xn,xc=arr["train"][0][trmask],arr["train"][1][trmask]; minority=ytr==1; majority=ytr==0
        nnew=max(0,int(majority.sum()/2-minority.sum())); sn,sc=tabddpm_augment(xn[minority],xc[minority],nnew,42,250)
        Xtr=np.r_[design["train"][trmask],np.c_[sn,enc.transform(sc)].astype(np.float32)]; ytr=np.r_[ytr,np.ones(nnew,int)]
        Xte=design["test"][temask]
    fttp=ft_probs(arr["test"][0][temask],arr["test"][1][temask],task); probabilities[task]["FT-Transformer"]=fttp
    rows.append({"task":task,"model":"FT-Transformer",**met(yte,fttp)})
    for name,model in fit_benchmarks(Xtr,ytr).items():
        p=model.predict_proba(Xte)[:,1]; probabilities[task][name]=p; rows.append({"task":task,"model":name,**met(yte,p)})
metrics=pd.DataFrame(rows); metrics.to_csv(OUT/"model_comparison_metrics.csv",index=False)

for task in ["HIV","ART"]:
    mask=np.ones(len(parts["test"]),bool) if task=="HIV" else parts["test"].ART_MONITORING_ELIGIBLE.eq(1).to_numpy()
    y=(parts["test"].HIV_SCREEN_LABEL.to_numpy(int) if task=="HIV" else parts["test"].loc[mask,"ART_UNSUPPRESSED_LABEL"].to_numpy(int))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for name,p in probabilities[task].items():
        fpr,tpr,_=roc_curve(y,p); pr,re,_=precision_recall_curve(y,p)
        axs[0].plot(fpr,tpr,label=f"{name} ({roc_auc_score(y,p):.3f})")
        axs[1].plot(re,pr,label=f"{name} ({average_precision_score(y,p):.3f})")
    axs[0].plot([0,1],[0,1],'k--',lw=1); axs[0].set(xlabel="False-positive rate",ylabel="Sensitivity",title=f"{task}: ROC curves")
    axs[1].axhline(y.mean(),color='k',ls='--',lw=1); axs[1].set(xlabel="Recall",ylabel="Precision",title=f"{task}: precision–recall curves")
    for ax in axs: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/f"{task.lower()}_roc_pr_curves.png",dpi=300); plt.close(fig)
    cm=metrics[(metrics.task==task)&(metrics.model=="FT-Transformer")].iloc[0]
    fig,ax=plt.subplots(figsize=(4.5,4)); sns.heatmap([[cm.tn,cm.fp],[cm.fn,cm.tp]],annot=True,fmt='.0f',cmap='Blues',cbar=False,ax=ax)
    ax.set(xlabel="Predicted",ylabel="Observed",xticklabels=["Negative","Positive"],yticklabels=["Negative","Positive"],title=f"{task}: FT-Transformer confusion matrix")
    fig.tight_layout(); fig.savefig(OUT/f"{task.lower()}_ft_confusion_matrix.png",dpi=300); plt.close(fig)

long=metrics.melt(id_vars=["task","model"],value_vars=["accuracy","sensitivity","specificity","precision","f1","auroc","auprc"],var_name="metric",value_name="value")
for task in ["HIV","ART"]:
    fig,ax=plt.subplots(figsize=(12,5)); sns.barplot(data=long[long.task==task],x="metric",y="value",hue="model",ax=ax)
    ax.set(ylim=(0,1.05),xlabel="",ylabel="Score",title=f"{task}: held-out test performance comparison"); ax.legend(fontsize=7,ncol=3)
    fig.tight_layout(); fig.savefig(OUT/f"{task.lower()}_metric_comparison.png",dpi=300); plt.close(fig)

hist=pd.read_csv(ROOT/"training_history.csv"); fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(hist.epoch,hist.train_loss,label="Training"); ax.plot(hist.epoch,hist.validation_loss,label="Validation")
best=int(hist.loc[hist.validation_loss.idxmin(),"epoch"]); ax.axvline(best,color='k',ls='--',label=f"Best epoch ({best})"); ax.set(xlabel="Epoch",ylabel="Multitask BCE loss",title="FT-Transformer learning curves"); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"ft_training_curves.png",dpi=300); plt.close(fig)

# Permutation SHAP on mixed FT inputs. Explain 60 deterministic test records against 20 training backgrounds.
feature_names=prep.num_cols+prep.cat_cols; nnum=len(prep.num_cols); bg_idx=np.linspace(0,len(parts["train"])-1,20,dtype=int)
background=np.c_[arr["train"][0][bg_idx],arr["train"][1][bg_idx]].astype(float)
lipid_protein=["TOTCHOL","HDL","TRIGLYC","NONHDL","MHLDLC","ALBU","PROT","GLOBULIN","A_G_RATIO","TC_HDL_RATIO","LDL_HDL_RATIO","TG_HDL_RATIO"]
shap_summary=[]
for task in ["HIV","ART"]:
    eligible=np.ones(len(parts["test"]),bool) if task=="HIV" else parts["test"].ART_MONITORING_ELIGIBLE.eq(1).to_numpy()
    ids=np.where(eligible)[0]; ids=ids[np.linspace(0,len(ids)-1,min(60,len(ids)),dtype=int)]
    explain_x=np.c_[arr["test"][0][ids],arr["test"][1][ids]].astype(float)
    def pred(z):
        zn=z[:,:nnum].astype(np.float32); zc=np.rint(z[:,nnum:]).astype(np.int64)
        for j,k in enumerate(prep.cardinalities): zc[:,j]=np.clip(zc[:,j],0,k-1)
        return ft_probs(zn,zc,task)
    explainer=shap.Explainer(pred,shap.maskers.Independent(background,max_samples=20),algorithm="permutation",feature_names=feature_names,seed=42)
    sv=explainer(explain_x,max_evals=2*len(feature_names)+1)
    keep=[feature_names.index(c) for c in lipid_protein]; bio=sv[:,keep]
    importance=np.abs(bio.values).mean(0); order=np.argsort(importance)[::-1]
    for j in order: shap_summary.append({"task":task,"feature":lipid_protein[j],"mean_abs_shap":importance[j]})
    shap.plots.bar(bio,max_display=12,show=False); plt.title(f"{task}: global lipid/protein importance"); plt.tight_layout(); plt.savefig(OUT/f"{task.lower()}_shap_global_bar.png",dpi=300,bbox_inches='tight'); plt.close()
    shap.plots.beeswarm(bio,max_display=12,show=False); plt.title(f"{task}: lipid/protein SHAP distribution"); plt.tight_layout(); plt.savefig(OUT/f"{task.lower()}_shap_beeswarm.png",dpi=300,bbox_inches='tight'); plt.close()
    p=pred(explain_x); local=int(np.argmax(p)); shap.plots.waterfall(bio[local],max_display=12,show=False)
    plt.title(f"{task}: patient-specific lipid/protein explanation\nCCSID {int(parts['test'].iloc[ids[local]].CCSID)}; probability {p[local]:.3f}")
    plt.tight_layout(); plt.savefig(OUT/f"{task.lower()}_shap_patient_waterfall.png",dpi=300,bbox_inches='tight'); plt.close()
pd.DataFrame(shap_summary).to_csv(OUT/"shap_lipid_protein_importance.csv",index=False)
print(metrics.to_string(index=False)); print('\nTop SHAP biomarkers\n',pd.DataFrame(shap_summary).groupby('task').head(5).to_string(index=False))
