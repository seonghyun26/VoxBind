#!/usr/bin/env python3
"""Build champion⊕partner (PCA-64) + partner-alone feature files on the COMMON shared pid set,
probe each with tta_6set (5-seed 0-4), parse r/rho/RMSE for all 6 cohorts, save JSON for the figure.
Partners: GET, EGNN, CheapNet, ProFSA (all frozen baseline encoders). CPU."""
import os, sys, csv, json, subprocess, re
import numpy as np, torch
REPO="/home/shpark/prj-denovo/VoxBind"; FEAT=f"{REPO}/voxbind/dataset/data/pdbbind/features"
OUT=f"{REPO}/notebook/html/260910/fig-ensemble"; TMP=f"{FEAT}/_figtmp"; os.makedirs(TMP, exist_ok=True)
CHAMP=f"{FEAT}/atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt"
COHORTS=["FULL","CL3","CL3-ID60","CL3-ID30","CASF-nontrain","CASF-clean"]

def load(path):
    d=torch.load(path, map_location='cpu', weights_only=False); f=d.get('features',d)
    return {k.lower(): np.asarray(v,np.float32) for k,v in f.items()}

champ=load(CHAMP)
get={**load(f"{REPO}/base/get/_edrscc/features/get_v2_seed0_graph_repr.pt"),
     **load(f"{REPO}/base/get/_casf_get/features/get_v2_seed0_casf_repr.pt")}
egnn=load(f"{REPO}/base/get/_edrscc/features/egnn_v2_seed0_all_repr.pt")
cheap=load(f"{REPO}/base/cheapnet/_edrscc/features/cheapnet_casf_seed0_prehead.pt")
prof={}
for s in ["repr_train_seed0","repr_valid_seed0","repr_lp_edrscc_v2_test_seed0","repr_casf_seed0"]:
    prof.update(load(f"{REPO}/base/profsa/_edrscc/features/{s}.pt"))
PARTNERS={"GET":get,"EGNN":egnn,"CheapNet":cheap,"ProFSA":prof}
split={r['pid'].lower():r['split'] for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2.csv"))}

# common shared pids: champion ∩ all partners
common=[p for p in champ if all(p.lower() in P for P in PARTNERS.values())]
print("common shared pids:", len(common), flush=True)
def z(X): m=X.mean(0,keepdims=True); s=X.std(0,keepdims=True); s[s==0]=1; return (X-m)/s
C=z(np.stack([champ[p] for p in common]))
tr=np.array([split.get(p.lower())=='train' for p in common])
def pca64(P):
    if P.shape[1]<=64: return z(P)
    mu=P[tr].mean(0); U,S,Vt=np.linalg.svd(P[tr]-mu, full_matrices=False)
    return z(((P-mu)@Vt[:64].T).astype(np.float32))
def save(name,M): torch.save({'features':{p:M[i] for i,p in enumerate(common)}}, f"{TMP}/{name}.pt")
save("base", C)

def probe(tta_name):
    cmd=["python","-u",f"{REPO}/voxbind/test/tta_6set_probe.py","--feat_base",f"{TMP}/base.pt",
         "--feat_tta",f"{TMP}/{tta_name}.pt","--seeds","5"]
    env=dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    out=subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=f"{REPO}/voxbind").stdout
    res={}
    for line in out.splitlines():
        parts=line.split()
        if parts and parts[0] in COHORTS and len(parts)>=9:
            c=parts[0]
            res[c]=dict(base_rho=float(parts[2].split('±')[0]), tta_rho=float(parts[3].split('±')[0]),
                        base_r=float(parts[5]), tta_r=float(parts[6]),
                        base_rmse=float(parts[7]), tta_rmse=float(parts[8]))
    return res

results={"champion":{}, "ensembles":{}, "alone":{}, "n_shared":len(common)}
for name,P in PARTNERS.items():
    Pv=np.stack([P[p.lower()] for p in common]); P64=pca64(Pv)
    save(f"ens_{name}", np.concatenate([C,P64],1))
    save(f"solo_{name}", P64)
    r_ens=probe(f"ens_{name}"); r_solo=probe(f"solo_{name}")
    if not results["champion"]:
        results["champion"]={c:{"r":v["base_r"],"rho":v["base_rho"],"rmse":v["base_rmse"]} for c,v in r_ens.items()}
    results["ensembles"][name]={c:{"r":v["tta_r"],"rho":v["tta_rho"],"rmse":v["tta_rmse"]} for c,v in r_ens.items()}
    results["alone"][name]={c:{"r":v["tta_r"],"rho":v["tta_rho"],"rmse":v["tta_rmse"]} for c,v in r_solo.items()}
    print(f"[{name}] ens ID30 rho={results['ensembles'][name].get('CL3-ID30',{}).get('rho')} solo={results['alone'][name].get('CL3-ID30',{}).get('rho')}", flush=True)

json.dump(results, open(f"{OUT}/ensemble_results.json","w"), indent=2)
print("saved", f"{OUT}/ensemble_results.json", flush=True)
