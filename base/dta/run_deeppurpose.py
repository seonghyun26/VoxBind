"""run_deeppurpose.py — train a DeepPurpose DTA model (DeepDTA / GraphDTA / MolTrans) on
lp_edrscc_v2 (seq+SMILES → pK), evaluate on v2 test + the 2019 holdout.

One model per invocation (so 3 can run on 3 GPUs). Dumps:
  base/_casf/{Method}_holdout2019_preds.csv   (pid,pred,y)  ← common94/compile pipeline
  base/dta/preds/{Method}_v2test_seed{S}.csv  (pid,pred,y)  ← Table-1 main + CL tiers

Usage: CUDA_VISIBLE_DEVICES=0 python run_deeppurpose.py --model deepdta --seed 0
"""
import argparse, csv, os
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data"
PREDS = HERE / "preds"; PREDS.mkdir(exist_ok=True)
CASF = REPO / "base" / "_casf"

# drug_encoding, target_encoding, display-name, train knobs
CONFIGS = {
    "deepdta":  ("CNN",         "CNN", "DeepDTA",  dict(cls_hidden_dims=[1024, 512], train_epoch=100,
                                                        LR=1e-3, batch_size=256)),
    # GraphDTA removed: dgl has no torch-2.7 wheel; dgl2.4-in-torch2.1 breaks DeepPurpose's API
    # (is_block), and the dgl-free MPNN drug encoder diverges to NaN at every LR. Left unimplemented.
    "moltrans": ("Transformer", "Transformer", "MolTrans", dict(cls_hidden_dims=[1024, 512],
                                                        train_epoch=100, LR=1e-4, batch_size=128)),
}


def load_csv(name):
    rows = list(csv.DictReader(open(DATA / f"{name}.csv")))
    return ([r["pid"] for r in rows], [r["smiles"] for r in rows],
            [r["seq"] for r in rows], np.array([float(r["pK"]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drug_enc", default=None, help="override drug encoder (e.g. DGL_GCN in the dgl env)")
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from DeepPurpose import DTI as models
    from DeepPurpose.utils import data_process, generate_config

    de, te, disp, knobs = CONFIGS[a.model]
    if a.drug_enc:
        de = a.drug_enc                       # faithful GraphDTA-GCN via the dgl env
    print(f"[{disp}] drug={de} target={te} seed={a.seed}")

    def proc(name):
        pid, smi, seq, y = load_csv(name)
        df = data_process(np.array(smi), np.array(seq), y, de, te,
                          split_method="no_split", random_seed=a.seed)
        return pid, df, y

    tr_pid, tr, _ = proc("v2_train")
    va_pid, va, _ = proc("v2_val")
    te_pid, ted, tey = proc("v2_test")
    ho_pid, hod, hoy = proc("holdout")
    cf_pid, cfd, cfy = proc("casf")

    config = generate_config(drug_encoding=de, target_encoding=te, **knobs)
    model = models.model_initialize(**config)
    model.train(tr, va, ted)          # early-stops on val internally

    def predict_dump(pid, df, y, out, tag):
        p = np.array(model.predict(df)).flatten()
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pid", "pred", "y"])
            for a_, b_, c_ in zip(pid, p, y):
                w.writerow([a_, float(b_), float(c_)])
        print(f"  {tag:8} r={pearsonr(p, y)[0]:.3f} rho={spearmanr(p, y).statistic:.3f} "
              f"rmse={np.sqrt(((p-y)**2).mean()):.3f} n={len(pid)}")
        return p

    predict_dump(te_pid, ted, tey, PREDS / f"{disp}_v2test_seed{a.seed}.csv", "v2-test")
    predict_dump(ho_pid, hod, hoy, CASF / f"{disp}_holdout2019_preds_seed{a.seed}.csv", "holdout")
    predict_dump(cf_pid, cfd, cfy, CASF / f"{disp}_casf2016_preds_seed{a.seed}.csv", "casf2016")


if __name__ == "__main__":
    main()
