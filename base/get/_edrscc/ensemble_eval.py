"""Per-seed + ensemble metrics for GET on a given test.pkl.
Usage: python ensemble_eval.py <test_pkl> <preds_seedA.jsonl> <preds_seedB.jsonl> ...
Reads gt from the preds (inference embeds correct neglog_aff). Ensemble = mean pred over seeds.
"""
import sys, json
import numpy as np
from scipy.stats import pearsonr, spearmanr

test_pkl = sys.argv[1]  # unused for gt (kept for provenance), gt read from preds
pred_files = sys.argv[2:]

def load(f):
    rows=[json.loads(l) for l in open(f)]
    return {r['id']:(r['label'], r['gt']) for r in rows}

def metrics(ids, y, yhat):
    y=np.array(y); yhat=np.array([0.0 if (np.isinf(v) or np.isnan(v)) else v for v in yhat])
    r=pearsonr(y,yhat)[0]; rho=spearmanr(y,yhat)[0]
    rmse=float(np.sqrt(((y-yhat)**2).mean())); mae=float(np.abs(y-yhat).mean())
    return dict(n=len(ids), pearson=float(r), spearman=float(rho), rmse=rmse, mae=mae)

seeds=[load(f) for f in pred_files]
ids=sorted(set.intersection(*[set(s) for s in seeds]))
gt=[seeds[0][i][1] for i in ids]

per=[]
for k,s in enumerate(seeds):
    m=metrics(ids, gt, [s[i][0] for i in ids])
    per.append(m)
    print(f"seed{k} ({pred_files[k].split('/')[-1]}): r={m['pearson']:.4f} rho={m['spearman']:.4f} rmse={m['rmse']:.4f}")

ens=[np.mean([s[i][0] for s in seeds]) for i in ids]
me=metrics(ids, gt, ens)
rs=[p['pearson'] for p in per]; rhos=[p['spearman'] for p in per]
print(f"MEAN±STD over {len(seeds)} seeds: r={np.mean(rs):.4f}±{np.std(rs):.4f} rho={np.mean(rhos):.4f}±{np.std(rhos):.4f}")
print(f"ENSEMBLE ({len(seeds)}-model): r={me['pearson']:.4f} rho={me['spearman']:.4f} rmse={me['rmse']:.4f} mae={me['mae']:.4f}  n={me['n']}")
json.dump(dict(per_seed=per, ensemble=me,
               mean_r=float(np.mean(rs)), std_r=float(np.std(rs)),
               mean_rho=float(np.mean(rhos)), std_rho=float(np.std(rhos))),
          open(sys.argv[1].replace('.pkl','')+'_ensemble_metrics.json','w'), indent=2)
