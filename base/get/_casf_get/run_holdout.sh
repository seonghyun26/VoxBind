set -e
PYTHON=/home/shpark/.conda/envs/get/bin/python
GETROOT=/home/shpark/prj-denovo/VoxBind/base/get
D=$GETROOT/_casf_get
PKL=$D/datasets/holdout2019/test.pkl
P=$D/preds
export CUDA_VISIBLE_DEVICES=4
cd $GETROOT
declare -A CK
CK[GET_0]=./_edrscc/models/GET/version_0/checkpoint/epoch19_step3480.ckpt
CK[GET_1]=./_edrscc/models/GET_seed1/version_1/checkpoint/epoch16_step2958.ckpt
CK[GET_2]=./_edrscc/models/GET_seed2/version_1/checkpoint/epoch19_step3480.ckpt
CK[EGNN_0]=./_edrscc/models/EGNN_v2/version_0/checkpoint/epoch17_step3132.ckpt
CK[EGNN_1]=./_edrscc/models/EGNN_v2_seed1/version_0/checkpoint/epoch16_step2958.ckpt
CK[EGNN_2]=./_edrscc/models/EGNN_v2_seed2/version_0/checkpoint/epoch11_step2088.ckpt
CK[EGNN_TD_0]=./_edrscc/models/EGNN_TD_v2/version_0/checkpoint/epoch6_step1218.ckpt
CK[EGNN_TD_1]=./_edrscc/models/EGNN_TD_v2_seed1/version_0/checkpoint/epoch11_step2088.ckpt
CK[EGNN_TD_2]=./_edrscc/models/EGNN_TD_v2_seed2/version_0/checkpoint/epoch11_step2088.ckpt
for M in GET EGNN EGNN_TD; do for S in 0 1 2; do
  echo "[$(date +%H:%M:%S)] $M seed$S"
  $PYTHON $D/run_casf_inference.py --ckpt "$GETROOT/${CK[${M}_${S}]}" --test_pkl $PKL --out $P/preds_${M}_holdout2019_seed${S}.jsonl --gpu 0
done; done
echo DONE_GET_FAMILY
