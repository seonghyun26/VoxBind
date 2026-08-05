#!/bin/bash
# Parallel two-tower affinity probe: extract tokens ONCE (shared disk cache), then fan out the
# seeds across GPUs (one process per seed), then aggregate mean±std. Each seed is independent
# (head init + data order), so this is result-identical to the sequential --seeds path, just
# spread over GPUs. Most useful when sweeping many head/spec conditions × seeds.
#
# usage: probe_twotower_parallel.sh <pocket_exp> <ligand_exp> <epoch> <seeds> <gpus_csv> [split]
#   e.g. probe_twotower_parallel.sh 260806_tt_pocket_protein_vdw_mc \
#            260806_tt_ligdens_protein_vdw_mc 49 3 0,1,2
set -u
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin/python
POCKET="$1"; LIGAND="$2"; EPOCH="$3"; SEEDS="$4"; GPUS="$5"; SPLIT="${6:-lp_edrscc_v2}"
IFS=',' read -ra GARR <<< "$GPUS"; NG=${#GARR[@]}
TAG="${POCKET}__${LIGAND}_e${EPOCH}"
OUTDIR="test/results/twotower_parallel/$TAG"
mkdir -p "$OUTDIR"
PROBE="test/twotower_probe.py"
COMMON=(--pocket_exp "$POCKET" --ligand_exp "$LIGAND" --epoch "$EPOCH" --split "$SPLIT")

echo "[1/3] extract tokens once on GPU ${GARR[0]} (shared cache) ..."
$PY "$PROBE" "${COMMON[@]}" --gpu "${GARR[0]}" --extract_only || { echo "extract failed"; exit 1; }

echo "[2/3] fan out $SEEDS seeds across GPUs [$GPUS] ..."
pids=()
for s in $(seq 0 $((SEEDS-1))); do
  g=${GARR[$((s % NG))]}
  $PY "$PROBE" "${COMMON[@]}" --gpu "$g" --seed "$s" --out "$OUTDIR/seed${s}.json" \
      > "$OUTDIR/seed${s}.log" 2>&1 &
  pids+=($!); echo "  seed $s -> GPU $g (pid $!)"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = "1" ] && echo "WARN: a seed worker failed — see $OUTDIR/seed*.log"

echo "[3/3] aggregate ..."
NO_WANDB="${NO_WANDB:-0}"
$PY - "$OUTDIR" "$SEEDS" "$POCKET" "$LIGAND" "$EPOCH" "$SPLIT" "$NO_WANDB" <<'PYEOF'
import json, sys, glob, numpy as np
outdir, seeds = sys.argv[1], int(sys.argv[2])
pocket, ligand, epoch, split, no_wandb = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7] == "1"
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{outdir}/seed*.json"))]
if not rows:
    print("no seed results found"); sys.exit(1)
print(f"\n=== two-tower parallel probe ({len(rows)}/{seeds} seeds) vs champion 0.644 ===")
agg = {"n": len(rows)}
for k in ("test_rho", "test_r", "test_rmse", "val_rho"):
    v = [r[k] for r in rows]
    agg[f"{k}_mean"] = float(np.mean(v)); agg[f"{k}_std"] = float(np.std(v))
    print(f"  {k:10s}: {np.mean(v):.4f} ± {np.std(v):.4f}")
json.dump(agg, open(f"{outdir}/aggregate.json", "w"), indent=2)
print(f"\n-> {outdir}/aggregate.json")
# log the aggregate as ONE clean wandb run (workers themselves skip wandb)
if not no_wandb:
    try:
        import wandb
        run = wandb.init(project="binding-affinity", entity="eddy26", job_type="probe", reinit=True,
                         name=f"twotower_{pocket}+{ligand}_e{epoch}_par",
                         tags=["probe", "twotower", "parallel", f"split:{split}", f"epoch:{epoch}"],
                         config=dict(kind="twotower_crossattn_parallel", pocket_exp=pocket,
                                     ligand_exp=ligand, epoch=int(epoch), split=split, n_seeds=len(rows)))
        run.summary.update(agg)
        run.finish()
        print("-> logged aggregate to wandb (eddy26/binding-affinity)")
    except Exception as e:
        print(f"[wandb] skipped ({e!r})")
PYEOF
