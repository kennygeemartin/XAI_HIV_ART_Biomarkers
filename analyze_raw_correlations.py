"""Correlation analysis of the untouched MWCCS proxy dataset."""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

DATA=Path('datasets/MWCCS_full_5844.csv'); OUT=Path('artifacts_preprocessed/results/raw_correlation_analysis'); OUT.mkdir(parents=True,exist_ok=True)
sns.set_theme(style='white',context='paper'); df=pd.read_csv(DATA)

# Raw clinical numbers only. IDs, dates, constants, explicit duplicate targets and masks are excluded.
numeric=["AGEATVIS","HEIGHT","WEIGHT","BMI","TOTCHOL","HDL","TRIGLYC","NONHDL","MHLDLC",
 "ALBU","PROT","GLOBULIN","A_G_RATIO","TC_HDL_RATIO","LDL_HDL_RATIO","TG_HDL_RATIO","HBA1C",
 "GLUCOSE","CREAT","AST","ALT","ART_DURATION_DAYS","VLOAD","CD4N","CD4P","CD8N","CD8P","HSRAT"]
targets=['HIV_SCREEN_LABEL','ART_UNSUPPRESSED_LABEL']
categorical=['SITE','STUDYHIST','VISTYPE','SEXBIRTH','FASTING','CHOLRX','NIACIN','FIBRATES','DIABETES','ONART','ARTV','HAARTV2','ADH']

pearson=df[numeric].corr(method='pearson',min_periods=30); spearman=df[numeric].corr(method='spearman',min_periods=30)
pearson.to_csv(OUT/'raw_numeric_pearson_matrix.csv'); spearman.to_csv(OUT/'raw_numeric_spearman_matrix.csv')
counts=pd.DataFrame(index=numeric,columns=numeric,dtype=int)
for a in numeric:
 for b in numeric: counts.loc[a,b]=int(df[[a,b]].dropna().shape[0])
counts.to_csv(OUT/'raw_numeric_pairwise_n.csv')

def top_pairs(corr,method):
 vals=[]
 for i,a in enumerate(corr.columns):
  for b in corr.columns[i+1:]: vals.append({'method':method,'variable_1':a,'variable_2':b,'correlation':corr.loc[a,b],'abs_correlation':abs(corr.loc[a,b]),'pairwise_n':int(counts.loc[a,b])})
 return pd.DataFrame(vals).sort_values('abs_correlation',ascending=False)
top=pd.concat([top_pairs(pearson,'Pearson'),top_pairs(spearman,'Spearman')]); top.to_csv(OUT/'ranked_numeric_correlation_pairs.csv',index=False)

# Raw point-biserial/Pearson correlations with binary outcomes. ART uses eligible observed outcomes only.
assoc=[]
for target in targets:
 sub=df if target=='HIV_SCREEN_LABEL' else df[df.ART_MONITORING_ELIGIBLE.eq(1)]
 for c in numeric:
  z=sub[[c,target]].dropna(); r=z[c].corr(z[target]) if len(z)>2 and z[c].nunique()>1 else np.nan
  assoc.append({'outcome':target,'variable':c,'correlation':r,'abs_correlation':abs(r),'n':len(z)})
target_corr=pd.DataFrame(assoc).sort_values(['outcome','abs_correlation'],ascending=[True,False]); target_corr.to_csv(OUT/'raw_numeric_target_correlations.csv',index=False)

def cramers_v(a,b):
 tab=pd.crosstab(a,b); n=tab.values.sum()
 if n==0 or min(tab.shape)<2:return np.nan
 chi2=chi2_contingency(tab,correction=False)[0]; phi2=chi2/n; r,k=tab.shape
 phi2c=max(0,phi2-(k-1)*(r-1)/(n-1)); rc=r-(r-1)**2/(n-1); kc=k-(k-1)**2/(n-1)
 return np.sqrt(phi2c/min(kc-1,rc-1))
cv=pd.DataFrame(index=categorical,columns=categorical,dtype=float)
for a in categorical:
 for b in categorical: cv.loc[a,b]=cramers_v(df[a].astype('string').fillna('Missing'),df[b].astype('string').fillna('Missing')) if a!=b else 1
cv.to_csv(OUT/'raw_categorical_cramers_v_matrix.csv')

# Numeric heatmaps.
for name,corr in [('pearson',pearson),('spearman',spearman)]:
 fig,ax=plt.subplots(figsize=(17,14)); mask=np.triu(np.ones_like(corr,dtype=bool),1)
 sns.heatmap(corr,mask=mask,cmap='vlag',center=0,vmin=-1,vmax=1,square=True,linewidths=.25,cbar_kws={'label':f'{name.title()} correlation'},ax=ax)
 ax.set_title(f'Raw numerical variables: {name.title()} correlation (pairwise complete observations)'); fig.tight_layout(); fig.savefig(OUT/f'raw_numeric_{name}_heatmap.png',dpi=300); plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(13,8))
for ax,target,title in zip(axs,targets,['HIV screening','ART unsuppression (eligible only)']):
 q=target_corr[target_corr.outcome.eq(target)].sort_values('correlation'); colors=np.where(q.correlation>=0,'#D55E00','#0072B2')
 ax.barh(q.variable,q.correlation,color=colors); ax.axvline(0,color='black',lw=.8); ax.set(xlim=(-1,1),xlabel='Raw point-biserial correlation',title=title)
fig.tight_layout(); fig.savefig(OUT/'raw_numeric_target_correlations.png',dpi=300); plt.close(fig)

fig,ax=plt.subplots(figsize=(11,9)); mask=np.triu(np.ones_like(cv,dtype=bool),1); sns.heatmap(cv,mask=mask,cmap='YlGnBu',vmin=0,vmax=1,square=True,annot=True,fmt='.2f',cbar_kws={'label':"Cramér's V"},ax=ax)
ax.set_title("Raw categorical-variable association (bias-corrected Cramér's V)"); fig.tight_layout(); fig.savefig(OUT/'raw_categorical_cramers_v_heatmap.png',dpi=300); plt.close(fig)

summary={'rows':len(df),'numeric_variables':len(numeric),'categorical_variables':len(categorical),'method_note':'No preprocessing; missing pairs omitted pairwise. ART outcome correlations restricted to eligible participants.'}
(OUT/'analysis_configuration.json').write_text(json.dumps(summary,indent=2))
print('TOP PEARSON\n',top[top.method.eq('Pearson')].head(15).to_string(index=False)); print('\nTARGETS\n',target_corr.groupby('outcome').head(10).to_string(index=False))
