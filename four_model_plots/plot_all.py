# -*- coding: utf-8 -*-
# BMI5101 public plotting code: real inputs must stay in an authorized local directory.
# No training, SHAP computation, uploads, or network calls. See README.md.
from pathlib import Path
import numpy as np,pandas as pd,json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score
# 所有路径相对于本脚本；不依赖原电脑路径。输出只写入新建的 output 文件夹。
import argparse
parser=argparse.ArgumentParser(description='Plot local four-model inputs; no training or network access.')
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=Path('output'))
args=parser.parse_args()
L=args.data_dir.resolve(); D=args.output_dir.resolve(); D.mkdir(parents=True,exist_ok=True)
F=D/'figures'; F.mkdir(exist_ok=True)
DEMO=(L/'SYNTHETIC_DEMO.json').is_file()
pred=pd.read_csv(L/'predictions.csv',float_precision='round_trip')
primary=['LR_B','RF_B','GBM_B_lasso','XGB_B']
colors={'LR_B':'#D55E00','RF_B':'#009E73','GBM_B_lasso':'#0072B2','XGB_B':'#CC79A7'}
names={'LR_B':'LR B','RF_B':'RF B + LASSO','GBM_B_lasso':'GBM B + LASSO','XGB_B':'XGBoost B'}
# 输入检查：每个方案训练994人、测试426人，结局为0/1，概率在0到1之间。
for model in primary:
 for cohort,n in [('Train apparent',994),('Test',426)]:
  q=pred[(pred.model==model)&(pred.cohort==cohort)]
  assert len(q)==n and q.row_index.is_unique, (model,cohort)
  assert q.observed.isin([0,1]).all() and q.p.between(0,1).all()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#555555','axes.axisbelow':True,'grid.color':'#e3e3e3','grid.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
plotrows=[]
def data(model,cohort):return pred[(pred.model==model)&pred.cohort.eq(cohort)].sort_values('row_index')
def panel(ax,letter,title):ax.set_title(title,pad=12);ax.text(-.12,1.05,letter,transform=ax.transAxes,fontweight='bold',fontsize=17);ax.grid(True)
def save(fig,name,pdf):
 if DEMO: fig.text(.5,.5,'SYNTHETIC DEMO - NOT STUDY RESULTS',ha='center',va='center',rotation=25,fontsize=24,color='red',alpha=.25)
 fig.savefig(F/(name+'.png'),dpi=260,bbox_inches='tight');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
def deciles(q):
 q=q.copy();q['bin']=pd.qcut(q.p,q=10,labels=False,duplicates='drop');rows=[];z=1.959963984540054
 for b,g in q.groupby('bin'):
  n=len(g);event=g.observed.mean();mean=g.p.mean();den=1+z*z/n;center=(event+z*z/(2*n))/den;delta=z*np.sqrt(event*(1-event)/n+z*z/(4*n*n))/den;rows.append(dict(bin=int(b),n=n,deaths=int(g.observed.sum()),mean_predicted=mean,observed=event,lower=center-delta,upper=center+delta))
 return pd.DataFrame(rows)
def dca(q,thresholds):
 y=q.observed.to_numpy();p=q.p.to_numpy();return np.array([np.mean((p>=t)&(y==1))-np.mean((p>=t)&(y==0))*t/(1-t) for t in thresholds])
with PdfPages(D/'Results_Fig4-6.pdf') as main:
 # Fig 4, four-panel layout mirrors paper, while clearly identifying apparent training.
 fig,ax=plt.subplots(2,2,figsize=(13,10));fig.subplots_adjust(top=.89,bottom=.12,left=.08,right=.97,hspace=.38,wspace=.28);fig.suptitle('Fig 4. Calibration and decision curves of four modelling strategies',fontsize=16,y=.975)
 fig.text(.5,.937,'Raw predictions; shared patient split. Training panels are apparent fit, not out-of-fold validation.',ha='center',fontsize=10)
 thresholds=np.round(np.arange(.05,.501,.005),3)
 for i,cohort in enumerate(['Train apparent','Test']):
  a,b=ax[i];label='Training apparent (n=994)' if i==0 else 'Held-out test (n=426)';panel(a,'AC'[i],label+' - calibration');panel(b,'BD'[i],label+' - DCA');a.plot([0,1],[0,1],ls='--',color='#777777',lw=1,label='Ideal')
  for m in primary:
   q=data(m,cohort);bins=deciles(q);a.errorbar(bins.mean_predicted,bins.observed,yerr=[np.maximum(0,bins.observed-bins.lower),np.maximum(0,bins.upper-bins.observed)],fmt='o-',color=colors[m],label=names[m],markersize=3.2,linewidth=1.25,elinewidth=.6,alpha=.85,capsize=0)
   for r in bins.to_dict('records'):plotrows.append(dict(figure='Fig4',kind='calibration',model=m,cohort=cohort,**r))
   nb=dca(q,thresholds);b.plot(thresholds,nb,color=colors[m],lw=1.7,label=names[m]);plotrows.extend(dict(figure='Fig4',kind='DCA',model=m,cohort=cohort,threshold=float(t),net_benefit=float(v)) for t,v in zip(thresholds,nb))
  prev=q.observed.mean();b.plot(thresholds,prev-(1-prev)*thresholds/(1-thresholds),color='#777777',lw=1.1,label='Treat all');b.axhline(0,color='black',ls=':',lw=1,label='Treat none');a.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted mortality risk',ylabel='Observed mortality proportion');b.set(xlim=(.05,.5),ylim=(-.05,.25),xlabel='Threshold probability',ylabel='Net benefit');a.legend(loc='upper left',frameon=False);b.legend(loc='upper right',frameon=False)
 fig.text(.08,.055,'Calibration: model-specific risk deciles with Wilson 95% intervals. Connected empirical rates; no fitted smoothing.',fontsize=9)
 fig.text(.08,.035,'DCA: TP/n - FP/n x threshold/(1-threshold); unstandardized net benefit. Exploratory hypothetical decisions.',fontsize=9)
 save(fig,'Fig4_四方案校准_DCA',main)
 # Fig 5: clearly defined SHAP importance rather than unnamed model-specific importance.
 # Fig5只需已核对的Top12汇总SHAP，无需模型对象。
 importance=pd.read_csv(L/'shap_importance.csv')
 tables={m:importance.loc[importance.model.eq(m),['feature','mean_abs_shap']].rename(columns={'mean_abs_shap':'importance'}) for m in primary}
 def clean(s):
  s=s.replace('num__','').replace('numeric__','').replace('cat__','').replace('categorical__','').replace('missingindicator_','Missing: ')
  for old,new in [('Medical/Surgical Intensive Care Unit (MICU/SICU)','MICU/SICU'),('SURGICAL SAME DAY ADMISSION','Surgical same-day'),('PHYSICIAN REFERRAL','Physician referral')]:s=s.replace(old,new)
  return '\n'.join(textwrap.wrap(s,38))
 fig,axs=plt.subplots(2,2,figsize=(14,11));fig.subplots_adjust(top=.89,bottom=.1,left=.24,right=.97,wspace=1.05,hspace=.48);fig.suptitle('Fig 5. Global SHAP importance within each modelling strategy',fontsize=16,y=.975);fig.text(.5,.94,'Top 12 predictors; RF grouped by raw field, other panels encoded features. Test n=426; independent scales.',ha='center',fontsize=10)
 out=[]
 for i,m in enumerate(primary):
  a=tables[m].sort_values('importance',ascending=False).head(12).iloc[::-1];ax=axs.flat[i];ax.barh(range(len(a)),a.importance,color=colors[m],height=.72);ax.set_yticks(range(len(a)),[clean(x) for x in a.feature],fontsize=8.5);ax.set_xlabel('Mean |SHAP| (probability)' if m=='RF_B' else 'Mean |SHAP| (log-odds)');ax.set_title(names[m],loc='left',pad=16);ax.text(-.12,1.07,'ABCD'[i],transform=ax.transAxes,fontweight='bold',fontsize=17);ax.grid(axis='x');ax.set_axisbelow(True)
  out.extend(dict(model=m,feature=r.feature,mean_abs_shap=r.importance,scale='probability' if m=='RF_B' else 'log-odds',n=426) for _,r in a.iloc[::-1].iterrows())
 fig.text(.08,.042,'LR, RF and GBM: supplied importance tables (synthetic when demo). XGBoost: supplied contribution arrays; no SHAP computation in this script.',fontsize=9);fig.text(.08,.023,'RF sums encoded contributions per raw field before averaging absolute values. Panel magnitudes are not comparable.',fontsize=9);save(fig,'Fig5_四方案SHAP重要性',main);pd.DataFrame(out).to_csv(D/'Fig5_作图数值.csv',index=False,encoding='utf-8-sig')
 # Fig 6: full-test XGBoost beeswarm and a single fixed, median-risk illustrative case.
 z=np.load(L/'XGB_B_SHAP.npz');sh=z['values'];feat=z['data'];fnames=z['names'];base=z['base'];probs=z['probability'];order=np.argsort(np.mean(abs(sh),axis=0))[::-1][:15]
 fig,(ax,bx)=plt.subplots(1,2,figsize=(16,8),gridspec_kw={'width_ratios':[1.1,1]});fig.subplots_adjust(top=.86,bottom=.15,left=.21,right=.97,wspace=.72);fig.suptitle('Fig 6. XGBoost B: global and individual SHAP explanations',fontsize=16,y=.975);fig.text(.5,.928,'Supplied SHAP arrays (synthetic when demo); n=426. Values and baseline are on the log-odds scale.',ha='center',fontsize=10)
 rng=np.random.default_rng(42)
 for row,j in enumerate(order[::-1]):
  vals=sh[:,j];v=feat[:,j];lo,hi=np.nanpercentile(v,[5,95]);scaled=np.clip((v-lo)/(hi-lo),0,1) if hi>lo else np.full(len(v),.5)
  # Density packing in narrow x bins, retaining each exact SHAP x coordinate.
  edges=np.linspace(vals.min()-1e-8,vals.max()+1e-8,51);bins=np.clip(np.digitize(vals,edges)-1,0,49);offset=np.zeros(len(vals))
  for bucket in np.unique(bins):
   idx=np.flatnonzero(bins==bucket);rng.shuffle(idx);n=len(idx);seq=np.arange(n)-(n-1)/2;offset[idx]=seq/(max(n,2)/2)*min(.36,.025*n)
  ax.scatter(vals,row+offset,c=scaled,cmap='inferno',vmin=0,vmax=1,s=10,alpha=.8,linewidths=0,rasterized=False)
 ax.axvline(0,color='#777777',lw=.8);ax.set_yticks(range(15),[clean(fnames[j]) for j in order[::-1]],fontsize=9);ax.set(xlabel='SHAP value (log-odds)',ylim=(-.7,14.7));ax.grid(axis='y',alpha=.4);ax.set_title('A  Global summary (n=426)',loc='left',pad=15,fontweight='bold');cbar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='inferno'),ax=ax,fraction=.035,pad=.025);cbar.set_ticks([0,1],labels=['Low','High']);cbar.set_label('Encoded feature value (within-feature scale)',fontsize=9)
 pos=int(np.argsort(probs,kind='mergesort')[len(probs)//2]);sv=sh[pos];top=np.argsort(abs(sv))[::-1][:8];rest=float(sv.sum()-sv[top].sum());contributions=list(sv[top])+[rest];labels=[clean(fnames[j]) for j in top]+[f'Other {len(sv)-8} features'];current=float(base[pos]);case_rows=[]
 for i,(label,contribution) in enumerate(zip(labels,contributions)):
  end=current+float(contribution);color='#D55E00' if contribution>=0 else '#0072B2';bx.barh(i,abs(contribution),left=min(current,end),color=color,height=.65);bx.plot([end,end],[i+.34,i+.66],color='#888888',lw=.8);bx.text(1.02,i,f'{contribution:+.3f}',transform=bx.get_yaxis_transform(),va='center',ha='left',fontsize=9);case_rows.append({'term':label.replace('\n',' '),'SHAP_log_odds':float(contribution)});current=end
 bx.invert_yaxis();bx.set_yticks(range(len(labels)),labels,fontsize=9);bx.axvline(float(base[pos]),color='#777777',ls='--',lw=1,label=f'Base log-odds {float(base[pos]):.3f}');bx.axvline(current,color='black',lw=1,label=f'Final log-odds {current:.3f}');bx.set_xlabel(f'Cumulative output (log-odds)\nBase {float(base[pos]):.3f}; final {current:.3f}');bx.grid(axis='x');bx.set_title('B  One median-risk test example',loc='left',pad=15,fontweight='bold');bx.text(.02,.99,f'Predicted mortality probability = {float(probs[pos]):.1%}',transform=bx.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
 fig.text(.08,.06,'A: horizontal positions are exact SHAP values; vertical packing is for visibility. Colours use transformed, encoded feature values.',fontsize=9);fig.text(.08,.036,'B: waterfall starts at the SHAP baseline and sums to the prediction. Case chosen by median predicted risk, without using its outcome.',fontsize=9);fig.text(.08,.014,'Explanations describe associations learned by this model, not treatment effects. Patient identifiers are omitted.',fontsize=9)
 save(fig,'Fig6_XGBoost蜂群与单例解释',main);pd.DataFrame(case_rows).to_csv(D/'Fig6_单例贡献_无患者ID.csv',index=False,encoding='utf-8-sig');(D/'Fig6_数值校验.json').write_text(json.dumps({'case_selection':'median predicted risk among 426 test patients; no outcome used','base_log_odds':float(base[pos]),'sum_shap':float(sv.sum()),'final_log_odds':float(current),'predicted_probability':float(probs[pos]),'sigmoid_of_sum':float(1/(1+np.exp(-current))),'n_explained':426},indent=2))
with PdfPages(D/'Results_补充图.pdf') as supp:
 # Supplement 1: apparent/test ROC, same four raw strategies.
 fig,axs=plt.subplots(1,2,figsize=(12,5.6));fig.subplots_adjust(top=.83,bottom=.17,wspace=.28);fig.suptitle('Supplement 1. ROC curves on the shared patient split',fontsize=15,y=.97)
 for i,cohort in enumerate(['Train apparent','Test']):
  ax=axs[i];panel(ax,'AB'[i],'Training apparent (n=994)' if i==0 else 'Held-out test (n=426)');ax.plot([0,1],[0,1],ls='--',color='#888888',lw=.9)
  for m in primary:
   q=data(m,cohort);fpr,tpr,_=roc_curve(q.observed,q.p);auc=roc_auc_score(q.observed,q.p);ax.plot(fpr,tpr,color=colors[m],lw=1.7,label=f'{names[m]}: {auc:.3f}');plotrows.extend(dict(figure='S1',kind='ROC',model=m,cohort=cohort,fpr=float(a),tpr=float(b)) for a,b in zip(fpr,tpr))
  ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 - specificity',ylabel='Sensitivity');ax.legend(loc='lower right',frameon=False)
 fig.text(.08,.04,'Training apparent curves are optimistic. They are not out-of-fold or external validation results.',fontsize=9);save(fig,'FigS1_ROC',supp)
 # Supplement 2: PR curves for positive-class performance.
 fig,ax=plt.subplots(figsize=(7.7,6.1));fig.subplots_adjust(top=.86,bottom=.16);fig.suptitle('Supplement 2. Precision-recall on the held-out test set',fontsize=14,y=.97)
 for m in primary:
  q=data(m,'Test');prec,rec,_=precision_recall_curve(q.observed,q.p);ap=average_precision_score(q.observed,q.p);ax.step(rec,prec,where='post',color=colors[m],lw=1.6,label=f'{names[m]}: AP {ap:.3f}');plotrows.extend(dict(figure='S2',kind='PR',model=m,cohort='Test',recall=float(a),precision=float(b)) for a,b in zip(rec,prec))
 ax.axhline(q.observed.mean(),color='#777777',ls='--',lw=1,label=f'Death prevalence {q.observed.mean():.3f}');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall (sensitivity)',ylabel='Precision (positive predictive value)');ax.grid(True);ax.legend(frameon=False,loc='upper right');fig.text(.1,.045,'AP = average precision; not trapezoidal PR-AUC. Positive outcome: in-hospital death.',fontsize=9);save(fig,'FigS2_PR',supp)
 # Supplement 3: RF raw and supplied calibration maps; no test-based method selection.
 fig,axs=plt.subplots(1,2,figsize=(12,5.7));fig.subplots_adjust(top=.84,bottom=.2,wspace=.3);fig.suptitle('Supplement 3. RF calibration methods on the held-out test set',fontsize=14,y=.97)
 for i,arm in enumerate(['A','B']):
  ax=axs[i];panel(ax,'AB'[i],'RF '+arm+' + LASSO');f=pd.read_csv(L/f'RF_{arm}_calibrated.csv');ax.plot([0,1],[0,1],color='#777777',ls='--',lw=.9)
  for col,color in [('raw','#666666'),('isotonic','#0072B2'),('temperature','#D55E00')]:
   q=f[['observed',col]].rename(columns={col:'p'});bins=deciles(q);brier=np.mean((q.p-q.observed)**2);ax.plot(bins.mean_predicted,bins.observed,'o-',color=color,lw=1.4,markersize=4,label=f'{col}: Brier {brier:.4f}');plotrows.extend(dict(figure='S3',kind='calibration',model='RF_'+arm+'_'+col,cohort='Test',**r) for r in bins.to_dict('records'))
  ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted mortality risk',ylabel='Observed mortality proportion');ax.legend(loc='upper left',frameon=False)
 fig.text(.08,.085,'Existing calibrators were fitted on training OOF probabilities; this analysis only applies them to saved-model predictions.',fontsize=9);fig.text(.08,.055,'Comparison is exploratory. Isotonic is not chosen as a primary method merely because its test Brier is lowest.',fontsize=9);save(fig,'FigS3_RF校准',supp)
pd.DataFrame(plotrows).to_csv(D/'曲线作图汇总数据.csv',index=False,encoding='utf-8-sig')
print('Main PDF: Fig4, Fig5, Fig6. Supplement PDF: ROC, PR, RF calibration. All six PNGs saved.',flush=True)
