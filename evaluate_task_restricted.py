"""Benchmarks and charts for the final task-restricted experiment."""
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,roc_curve,precision_recall_curve
import train_ft_transformer_preprocessed as pp
from train_task_restricted_ft import TaskRestrictedFT,COMMON_NUM,ART_NUM,COMMON_CAT,ART_CAT,ALL_NUM,ALL_CAT

ROOT=Path('artifacts_task_restricted'); OUT=ROOT/'results'; OUT.mkdir(exist_ok=True); sns.set_theme(style='whitegrid',context='paper')
pp.NUMERICAL=ALL_NUM; pp.CATEGORICAL=ALL_CAT
s=joblib.load(ROOT/'preprocessor.joblib'); prep=pp.StructuredPreprocessor()
for k,v in s.items():setattr(prep,k,v)
df=pd.read_csv('datasets/MWCCS_full_5844.csv'); parts={x:df[df.DATA_SPLIT.eq(x)].reset_index(drop=True) for x in ['train','test']}; arr={x:prep.transform(d) for x,d in parts.items()}
enc=OneHotEncoder(categories=[np.arange(k) for k in prep.cardinalities],handle_unknown='ignore',sparse_output=False).fit(arr['train'][1])
ck=torch.load(ROOT/'best_model.pt',map_location='cpu',weights_only=False); ft=TaskRestrictedFT(prep.cardinalities); ft.load_state_dict(ck['model_state']); ft.eval()
@torch.no_grad()
def ftp(xn,xc,t):return torch.sigmoid(ft(torch.tensor(xn),torch.tensor(xc),t.lower())).numpy()
def metrics(y,p):
 q=p>=.5; tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel(); return {'accuracy':accuracy_score(y,q),'sensitivity':recall_score(y,q),'specificity':tn/(tn+fp),'precision':precision_score(y,q),'f1':f1_score(y,q),'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'tn':tn,'fp':fp,'fn':fn,'tp':tp}
def models(y):
 w=(y==0).sum()/(y==1).sum(); return {'Logistic regression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42),'Random forest':RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),'XGBoost':XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,scale_pos_weight=w,n_jobs=-1,random_state=42),'MLP':MLPClassifier(hidden_layer_sizes=(128,64),batch_size=64,max_iter=300,early_stopping=True,n_iter_no_change=10,random_state=42)}
rows=[]; probs={}
for task in ['HIV','ART']:
 if task=='HIV':
  ni=len(COMMON_NUM); ci=len(COMMON_CAT); Xtr=np.c_[arr['train'][0][:,:ni],enc.transform(arr['train'][1])[:,:sum(prep.cardinalities[:ci])]]; Xte=np.c_[arr['test'][0][:,:ni],enc.transform(arr['test'][1])[:,:sum(prep.cardinalities[:ci])]]; ytr=parts['train'].HIV_SCREEN_LABEL.to_numpy(int); yte=parts['test'].HIV_SCREEN_LABEL.to_numpy(int); mask=np.ones(len(yte),bool)
 else:
  tr=parts['train'].ART_MONITORING_ELIGIBLE.eq(1).to_numpy(); mask=parts['test'].ART_MONITORING_ELIGIBLE.eq(1).to_numpy(); ytr=parts['train'].loc[tr,'ART_UNSUPPRESSED_LABEL'].to_numpy(int); yte=parts['test'].loc[mask,'ART_UNSUPPRESSED_LABEL'].to_numpy(int)
  xn,xc=arr['train'][0][tr],arr['train'][1][tr]; mi=ytr==1; ma=ytr==0; n=int(ma.sum()/2-mi.sum()); sn,sc=pp.tabddpm_augment(xn[mi],xc[mi],n,42,250)
  Xtr=np.r_[np.c_[xn,enc.transform(xc)],np.c_[sn,enc.transform(sc)]]; ytr=np.r_[ytr,np.ones(n,int)]; Xte=np.c_[arr['test'][0][mask],enc.transform(arr['test'][1][mask])]
 p=ftp(arr['test'][0][mask],arr['test'][1][mask],task); probs[task]={'FT-Transformer':p}; rows.append({'task':task,'model':'FT-Transformer',**metrics(yte,p)})
 for name,m in models(ytr).items():m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; probs[task][name]=p; rows.append({'task':task,'model':name,**metrics(yte,p)})
metrics_df=pd.DataFrame(rows); metrics_df.to_csv(OUT/'task_restricted_model_comparison.csv',index=False)
for task in ['HIV','ART']:
 mask=np.ones(len(parts['test']),bool) if task=='HIV' else parts['test'].ART_MONITORING_ELIGIBLE.eq(1).to_numpy(); y=parts['test'].HIV_SCREEN_LABEL.to_numpy(int) if task=='HIV' else parts['test'].loc[mask,'ART_UNSUPPRESSED_LABEL'].to_numpy(int)
 fig,axs=plt.subplots(1,2,figsize=(11,4.5))
 for name,p in probs[task].items():
  fpr,tpr,_=roc_curve(y,p); pr,re,_=precision_recall_curve(y,p); axs[0].plot(fpr,tpr,label=f'{name} ({roc_auc_score(y,p):.3f})'); axs[1].plot(re,pr,label=f'{name} ({average_precision_score(y,p):.3f})')
 axs[0].plot([0,1],[0,1],'k--'); axs[1].axhline(y.mean(),color='k',ls='--'); axs[0].set(xlabel='False-positive rate',ylabel='Sensitivity',title=f'{task}: ROC'); axs[1].set(xlabel='Recall',ylabel='Precision',title=f'{task}: precision–recall')
 for ax in axs:ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(OUT/f'{task.lower()}_roc_pr.png',dpi=300);plt.close(fig)
 q=metrics_df[metrics_df.task.eq(task)].melt(id_vars=['task','model'],value_vars=['accuracy','sensitivity','specificity','precision','f1','auroc','auprc'],var_name='metric',value_name='value');fig,ax=plt.subplots(figsize=(12,5));sns.barplot(q,x='metric',y='value',hue='model',ax=ax);ax.set(ylim=(0,1.05),title=f'{task}: task-restricted held-out performance',xlabel='',ylabel='Score');fig.tight_layout();fig.savefig(OUT/f'{task.lower()}_metrics.png',dpi=300);plt.close(fig)
hist=pd.read_csv(ROOT/'training_history.csv');fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(hist.epoch,hist.train_loss,label='Training');ax.plot(hist.epoch,hist.validation_loss,label='Validation');ax.axvline(hist.loc[hist.validation_loss.idxmin(),'epoch'],ls='--',color='k',label='Selected checkpoint');ax.set(xlabel='Epoch',ylabel='Multitask BCE loss',title='Task-restricted FT-Transformer learning curves');ax.legend();fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=300);plt.close(fig)
print(metrics_df.to_string(index=False))
