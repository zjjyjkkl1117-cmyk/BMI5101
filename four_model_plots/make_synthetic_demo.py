"""Generate artificial fixtures from a fixed RNG seed; never reads patient data."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('demo_data'));args=p.parse_args()
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise SystemExit('Choose an empty output directory; refusing to overwrite inputs.')
rng=np.random.default_rng(2026)
models=['LR_B','RF_B','GBM_B_lasso','XGB_B']
names=np.array([f'feature_{i:02d}' for i in range(1,25)])
def sigmoid(x):return 1/(1+np.exp(-x))
records=[]
for cohort,n in [('Train apparent',994),('Test',426)]:
    x=rng.normal(size=(n,24));contrib=x*np.linspace(.22,.02,24)
    probability=sigmoid(-1.4+contrib.sum(axis=1));y=rng.binomial(1,probability)
    for i,model in enumerate(models):
        prob=probability if model=='XGB_B' else sigmoid(-1.4+contrib.sum(axis=1)+rng.normal(0,.2,n))
        records.append(pd.DataFrame(dict(model=model,cohort=cohort,observed=y,p=prob,row_index=np.arange(n))))
        if cohort=='Test' and model=='RF_B':rf_p=prob
    if cohort=='Test':
        np.savez_compressed(out/'XGB_B_SHAP.npz',values=contrib,base=np.full(n,-1.4),data=x,names=names,probability=probability)
        for v in ['A','B']:
            pd.DataFrame(dict(observed=y,raw=rf_p,isotonic=np.clip(rf_p*.9,0,1),temperature=sigmoid(np.log(rf_p/(1-rf_p))/.9))).to_csv(out/f'RF_{v}_calibrated.csv',index=False)
pd.concat(records).to_csv(out/'predictions.csv',index=False)
importance=[]
for model in models:
    for j in range(12):importance.append(dict(model=model,feature=names[j],mean_abs_shap=float(.2/(j+1)),scale='probability' if model=='RF_B' else 'log-odds',n=426))
pd.DataFrame(importance).to_csv(out/'shap_importance.csv',index=False)
(out/'SYNTHETIC_DEMO.json').write_text(json.dumps({'synthetic':True,'seed':2026,'purpose':'Software demonstration only; not study findings; no patient data read.'}))
print('Synthetic demo inputs created:',out)
