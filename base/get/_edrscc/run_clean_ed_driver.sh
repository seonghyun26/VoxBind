#!/bin/bash
# Full GET clean_ed_v1 pipeline: wait data -> build indep test -> train 3 seeds -> infer+eval both tests.
set -e
cd /home/shpark/prj-denovo/VoxBind/base/get
source $(conda info --base)/etc/profile.d/conda.sh; conda activate get
LOG=_edrscc/logs
DS=datasets/clean_ed_v1

echo "[driver] waiting for make_data (clean_ed_v1) ..."
while pgrep -f "make_data.py --split_csv.*clean_ed_v1" >/dev/null; do sleep 15; done
[ -f $DS/train.pkl ] && [ -f $DS/valid.pkl ] && [ -f $DS/test.pkl ] || { echo "[driver] FATAL: data missing"; exit 1; }
echo "[driver] data ready: $(python -c "import pickle;print({k:len(pickle.load(open('$DS/'+k+'.pkl','rb'))) for k in ['train','valid','test']})")"

# build indep test.pkl by filtering clean_ed_v1 test to indep test pids
python - <<'PYIN'
import pickle, pandas as pd, os
sp=pd.read_csv("/home/shpark/prj-denovo/VoxBind/voxbind/splits/clean_ed_v1_indep.csv")
sp["pid"]=sp["pid"].astype(str).str.lower()
indep=set(sp.loc[sp["split"]=="test","pid"])
test=pickle.load(open("datasets/clean_ed_v1/test.pkl","rb"))
sub=[t for t in test if t["id"].lower() in indep]
pickle.dump(sub, open("datasets/clean_ed_v1/test_indep.pkl","wb"))
print(f"[driver] indep test.pkl: {len(sub)} / target {len(indep)}")
PYIN

# launch 3 seed trainings on GPUs 0,1,2
declare -A CFG=( [0]=clean_ed_v1_get.json [1]=clean_ed_v1_get_seed1.json [2]=clean_ed_v1_get_seed2.json )
declare -A PORT=( [0]=9931 [1]=9932 [2]=9933 )
for g in 0 1 2; do
  echo "[driver] launch seed$g on GPU $g cfg ${CFG[$g]}"
  GPU=$g PORT=${PORT[$g]} bash scripts/train/train.sh _edrscc/${CFG[$g]} > $LOG/train_clean_ed_seed$g.log 2>&1 &
done
echo "[driver] waiting for 3 seed trainings ..."
wait
echo "[driver] ALL_TRAIN_DONE"

# inference + eval per seed on both test sets
# restrict to GPUs 0-3 (avoid 4-7 per user pref)
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F',' '$1+0<=3' | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
declare -A SD=( [0]=GET_clean_ed_v1 [1]=GET_clean_ed_v1_seed1 [2]=GET_clean_ed_v1_seed2 )
PREDS_MAIN=""; PREDS_INDEP=""
for g in 0 1 2; do
  CKPT=$(grep -oE "Validation: [0-9.]+, save path: [^ ]+\.ckpt" $LOG/train_clean_ed_seed$g.log | sed -E 's/Validation: ([0-9.]+), save path: (.*)/\1 \2/' | sort -n | head -1 | awk '{print $2}')
  echo "[driver] seed$g best ckpt: $CKPT"
  CUDA_VISIBLE_DEVICES=$GPU python inference.py --test_set datasets/clean_ed_v1/test.pkl --task PDBBind \
      --ckpt "$CKPT" --gpu 0 --save_path _edrscc/preds_clean_ed_seed$g.jsonl 2>/dev/null
  CUDA_VISIBLE_DEVICES=$GPU python inference.py --test_set datasets/clean_ed_v1/test_indep.pkl --task PDBBind \
      --ckpt "$CKPT" --gpu 0 --save_path _edrscc/preds_clean_ed_indep_seed$g.jsonl 2>/dev/null
  PREDS_MAIN="$PREDS_MAIN _edrscc/preds_clean_ed_seed$g.jsonl"
  PREDS_INDEP="$PREDS_INDEP _edrscc/preds_clean_ed_indep_seed$g.jsonl"
done

echo "========== GET clean_ed_v1 (CASF-2016, 214) =========="
python _edrscc/ensemble_eval.py datasets/clean_ed_v1/test.pkl $PREDS_MAIN
echo "========== GET clean_ed_v1_indep (109) =========="
python _edrscc/ensemble_eval.py datasets/clean_ed_v1/test_indep.pkl $PREDS_INDEP
echo "[driver] DONE"
