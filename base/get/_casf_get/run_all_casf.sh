#!/usr/bin/env bash
# run_all_casf.sh — run CASF-2016 inference for GET, EGNN, EGNN_TD (3 seeds each)
# then score and write base/_casf/{GET,EGNN,EGNN_TD}.json
# GPU: 4 (CUDA_VISIBLE_DEVICES=4 → model sees it as gpu=0)
set -e
PYTHON=/home/shpark/.conda/envs/get/bin/python
GETROOT=/home/shpark/prj-denovo/VoxBind/base/get
CASF_GET_DIR=$GETROOT/_casf_get
CASF_PKL=$CASF_GET_DIR/datasets/casf2016/test.pkl
MANIFEST=$CASF_GET_DIR/datasets/casf2016/manifest.csv
PREDS_DIR=$CASF_GET_DIR/preds
CASF_OUT=/home/shpark/prj-denovo/VoxBind/base/_casf
mkdir -p $PREDS_DIR

export CUDA_VISIBLE_DEVICES=4

cd $GETROOT

echo "======= GET ======="
for SEED in 0 1 2; do
  case $SEED in
    0) CKPT="./_edrscc/models/GET/version_0/checkpoint/epoch19_step3480.ckpt" ;;
    1) CKPT="./_edrscc/models/GET_seed1/version_1/checkpoint/epoch16_step2958.ckpt" ;;
    2) CKPT="./_edrscc/models/GET_seed2/version_1/checkpoint/epoch19_step3480.ckpt" ;;
  esac
  OUT=$PREDS_DIR/preds_GET_casf_seed${SEED}.jsonl
  echo "  seed $SEED: $CKPT -> $OUT"
  $PYTHON $CASF_GET_DIR/run_casf_inference.py \
    --ckpt "$GETROOT/$CKPT" \
    --test_pkl $CASF_PKL \
    --out $OUT \
    --gpu 0
done

echo "======= EGNN ======="
for SEED in 0 1 2; do
  case $SEED in
    0) CKPT="./_edrscc/models/EGNN_v2/version_0/checkpoint/epoch17_step3132.ckpt" ;;
    1) CKPT="./_edrscc/models/EGNN_v2_seed1/version_0/checkpoint/epoch16_step2958.ckpt" ;;
    2) CKPT="./_edrscc/models/EGNN_v2_seed2/version_0/checkpoint/epoch11_step2088.ckpt" ;;
  esac
  OUT=$PREDS_DIR/preds_EGNN_casf_seed${SEED}.jsonl
  echo "  seed $SEED: $CKPT -> $OUT"
  $PYTHON $CASF_GET_DIR/run_casf_inference.py \
    --ckpt "$GETROOT/$CKPT" \
    --test_pkl $CASF_PKL \
    --out $OUT \
    --gpu 0
done

echo "======= EGNN_TD ======="
for SEED in 0 1 2; do
  case $SEED in
    0) CKPT="./_edrscc/models/EGNN_TD_v2/version_0/checkpoint/epoch6_step1218.ckpt" ;;
    1) CKPT="./_edrscc/models/EGNN_TD_v2_seed1/version_0/checkpoint/epoch11_step2088.ckpt" ;;
    2) CKPT="./_edrscc/models/EGNN_TD_v2_seed2/version_0/checkpoint/epoch11_step2088.ckpt" ;;
  esac
  OUT=$PREDS_DIR/preds_EGNN_TD_casf_seed${SEED}.jsonl
  echo "  seed $SEED: $CKPT -> $OUT"
  $PYTHON $CASF_GET_DIR/run_casf_inference.py \
    --ckpt "$GETROOT/$CKPT" \
    --test_pkl $CASF_PKL \
    --out $OUT \
    --gpu 0
done

echo "======= Scoring ======="
$PYTHON $CASF_GET_DIR/score_casf.py \
  --model GET \
  --preds $PREDS_DIR/preds_GET_casf_seed0.jsonl $PREDS_DIR/preds_GET_casf_seed1.jsonl $PREDS_DIR/preds_GET_casf_seed2.jsonl \
  --manifest $MANIFEST \
  --out_json $CASF_OUT/GET.json

$PYTHON $CASF_GET_DIR/score_casf.py \
  --model EGNN \
  --preds $PREDS_DIR/preds_EGNN_casf_seed0.jsonl $PREDS_DIR/preds_EGNN_casf_seed1.jsonl $PREDS_DIR/preds_EGNN_casf_seed2.jsonl \
  --manifest $MANIFEST \
  --out_json $CASF_OUT/EGNN.json

$PYTHON $CASF_GET_DIR/score_casf.py \
  --model EGNN_TD \
  --preds $PREDS_DIR/preds_EGNN_TD_casf_seed0.jsonl $PREDS_DIR/preds_EGNN_TD_casf_seed1.jsonl $PREDS_DIR/preds_EGNN_TD_casf_seed2.jsonl \
  --manifest $MANIFEST \
  --out_json $CASF_OUT/EGNN_TD.json

echo "Done! Results in $CASF_OUT/"
