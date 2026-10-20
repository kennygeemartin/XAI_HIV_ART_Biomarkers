"""Create before/after preprocessing tables and charts for the MWCCS proxy."""
from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_ft_transformer_preprocessed import StructuredPreprocessor

ART=Path('artifacts_preprocessed'); OUT=ART/'results'/'data_configuration'; OUT.mkdir(parents=True,exist_ok=True)
sns.set_theme(style='whitegrid',context='paper'); df=pd.read_csv('datasets/MWCCS_full_5844.csv')
state=joblib.load(ART/'preprocessor.joblib'); prep=StructuredPreprocessor()
for k,v in state.items(): setattr(prep,k,v)
parts={s:df[df.DATA_SPLIT.eq(s)].reset_index(drop=True) for s in ['train','validation','test']}

# Overall dataset and fixed split composition.
rows=[]
for split,d in [('Overall',df),('Train',parts['train']),('Validation',parts['validation']),('Test',parts['test'])]:
    eligible=d.ART_MONITORING_ELIGIBLE.eq(1); art=d.loc[eligible,'ART_UNSUPPRESSED_LABEL']
    rows.append({'partition':split,'participants':len(d),'percent_dataset':100*len(d)/len(df),
      'hiv_negative':int((d.HIV_SCREEN_LABEL==0).sum()),'hiv_positive':int((d.HIV_SCREEN_LABEL==1).sum()),
      'hiv_positive_percent':100*d.HIV_SCREEN_LABEL.mean(),'art_eligible':int(eligible.sum()),
      'art_suppressed':int((art==0).sum()),'art_unsuppressed':int((art==1).sum()),
      'art_unsuppressed_percent':100*art.mean() if len(art) else np.nan})
composition=pd.DataFrame(rows); composition.to_csv(OUT/'dataset_composition.csv',index=False)

# Variable-level before/after configuration.
report=json.load(open(ART/'preprocessing_report.json')); selected=set(prep.num_cols+prep.cat_cols)
roles={c:('Numerical predictor' if c in prep.num_cols else 'Categorical predictor' if c in prep.cat_cols else
          'Outcome/mask' if c in ['HIV_SCREEN_LABEL','ART_UNSUPPRESSED_LABEL','ART_SUPPRESSION_LABEL','SCREENING_ELIGIBLE','ART_MONITORING_ELIGIBLE'] else
          'Excluded leakage' if c in ['HIVPOS','VLOAD','UD_VL','NS_VL','LLQ_VL'] else 'Excluded metadata/other') for c in df.columns}
features=[]
for c in df.columns:
    features.append({'variable':c,'role_after_preprocessing':roles[c],'original_dtype':str(df[c].dtype),
      'overall_missing_n':int(df[c].isna().sum()),'overall_missing_percent':100*df[c].isna().mean(),
      'training_missing_percent':100*parts['train'][c].isna().mean(),'retained_as_predictor':c in selected,
      'operation':('log1p + iterative imputation + Z-score' if c in prep.log_cols else
                   'iterative imputation + Z-score' if c in prep.num_cols else
                   'mode imputation + embedding token' if c in prep.cat_cols else 'excluded')})
feature_table=pd.DataFrame(features); feature_table.to_csv(OUT/'variable_configuration_before_after.csv',index=False)

summary=pd.DataFrame([
 {'stage':'Raw dataset','rows':len(df),'predictor_features':len(df.columns)-5,'numerical_tokens':0,'categorical_tokens':0,'missing_predictor_cells':int(df.isna().sum().sum()),'synthetic_augmented_rows':0},
 {'stage':'After cleaning and feature selection','rows':len(df),'predictor_features':len(selected),'numerical_tokens':len(prep.num_cols),'categorical_tokens':len(prep.cat_cols),'missing_predictor_cells':int(df[list(selected)].isna().sum().sum()),'synthetic_augmented_rows':0},
 {'stage':'After imputation/transformation','rows':len(df),'predictor_features':len(selected),'numerical_tokens':len(prep.num_cols),'categorical_tokens':len(prep.cat_cols),'missing_predictor_cells':0,'synthetic_augmented_rows':0},
 {'stage':'FT training set after ART augmentation','rows':len(parts['train'])+828,'predictor_features':len(selected),'numerical_tokens':len(prep.num_cols),'categorical_tokens':len(prep.cat_cols),'missing_predictor_cells':0,'synthetic_augmented_rows':828},
]); summary.to_csv(OUT/'preprocessing_stage_summary.csv',index=False)

# Chart 1: partition and target composition.
fig,axs=plt.subplots(1,3,figsize=(14,4.5)); split=composition.iloc[1:]
axs[0].bar(split.partition,split.participants,color=['#4C78A8','#F58518','#54A24B']); axs[0].set(title='Participant-level partitions',ylabel='Participants')
for label,color in [('hiv_negative','#9ecae1'),('hiv_positive','#de2d26')]: axs[1].bar(split.partition,split[label],bottom=split['hiv_negative'] if label=='hiv_positive' else None,label=label.replace('_',' ').title(),color=color)
axs[1].set(title='HIV screening classes',ylabel='Participants'); axs[1].legend()
for label,color in [('art_suppressed','#74c476'),('art_unsuppressed','#756bb1')]: axs[2].bar(split.partition,split[label],bottom=split['art_suppressed'] if label=='art_unsuppressed' else None,label=label.replace('_',' ').title(),color=color)
axs[2].set(title='ART classes among eligible participants',ylabel='Participants'); axs[2].legend(); fig.tight_layout(); fig.savefig(OUT/'dataset_and_class_composition.png',dpi=300); plt.close(fig)

# Chart 2: selected feature missingness before and after.
miss=feature_table[feature_table.retained_as_predictor].sort_values('training_missing_percent',ascending=True)
fig,ax=plt.subplots(figsize=(9,10)); ax.barh(miss.variable,miss.training_missing_percent,label='Before preprocessing',color='#E45756'); ax.barh(miss.variable,np.zeros(len(miss)),label='After imputation',color='#72B7B2')
ax.set(xlabel='Missing values in training partition (%)',title='Selected-feature missingness before and after preprocessing'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'missingness_before_after.png',dpi=300); plt.close(fig)

# Chart 3: selected distributions before and after standardized transformation.
chosen=['TOTCHOL','HDL','TRIGLYC','ALBU','PROT','GLOBULIN','BMI','ART_DURATION_DAYS']; xn,_=prep.transform(parts['train'])
fig,axs=plt.subplots(2,4,figsize=(15,7));
for ax,c in zip(axs.ravel(),chosen):
    raw=parts['train'][c].dropna(); after=xn[:,prep.num_cols.index(c)]; ax.hist(raw,bins=25,density=True,alpha=.55,label='Raw'); ax2=ax.twiny(); ax2.hist(after,bins=25,density=True,alpha=.35,color='#E45756',label='Processed Z-score'); ax.set_title(c); ax.set_xlabel('Raw scale'); ax2.set_xlabel('Processed scale')
fig.tight_layout(); fig.savefig(OUT/'selected_distributions_before_after.png',dpi=300); plt.close(fig)

# Chart 4: ART augmentation.
aug=json.load(open(ART/'augmentation_report.json')); before=[aug['eligible_majority'],aug['eligible_minority']]; after=[aug['eligible_majority'],aug['eligible_minority']+aug['synthetic_minority_rows']]
x=np.arange(2); width=.35; fig,ax=plt.subplots(figsize=(7,4.5)); ax.bar(x-width/2,before,width,label='Before augmentation'); ax.bar(x+width/2,after,width,label='After training-only augmentation'); ax.set_xticks(x,['Suppressed','Unsuppressed']); ax.set(ylabel='Eligible training records',title='ART training-class composition before and after TabDDPM-style augmentation'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'art_augmentation_before_after.png',dpi=300); plt.close(fig)
print(composition.to_string(index=False)); print('\n',summary.to_string(index=False))
