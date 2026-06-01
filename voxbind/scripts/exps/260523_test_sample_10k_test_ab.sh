#!/bin/bash
# 09_sample_10k_test_ab.sh — full test-split sampling for the 10k X-ray density A/B.
#
# Samples the 10k-subset density / noise VoxBind models (both epoch 99) on the
# full CrossDocked TEST split (100 pockets). 8 pockets have no x-ray map, so
# sample.py auto-skips them for these density-conditioned models -> 92 pockets
# sampled, the SAME 92 for both arms (aligned evaluation). 10 samples/pocket.
#
# One arm per GPU, in parallel (GPUs 4-7 are busy with the 10k baseline):
#     density  -> GPU 2   real x-ray density   (xray_crops/test)
#     noise    -> GPU 3   noise density        (xray_crops_noise/test)
#
# The 10k noise run trained on xray_crops_noise_10k/, which has no test split;
# the noise test crops live in xray_crops_noise/test (92 noise crops, same 92
# indices as xray_crops/test) — content-free, so reusing them is correct.
#
# Outputs: exps/<run>/samples/res_ep99_test/target_NN/{samples.sdf,...}
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

# Prereqs: both test crop dirs must exist (92 crops each, aligned indices).
for d in "$DATA/xray_crops/test" "$DATA/xray_crops_noise/test"; do
    if [ ! -d "$d" ]; then
        echo "[10k-test $(ts)] ERROR: $d missing"
        exit 1
    fi
done

echo "[10k-test $(ts)] full test-split sampling — density(GPU2) / noise(GPU3), ep99, 10/pocket"

# $1=name  $2=exp dir  $3=crops_dir  $4=gpu
sample_arm () {
    local name=$1 exp=$2 crops=$3 gpu=$4
    local out=$VOX/exps/$exp/samples/res_ep99_test
    echo "[10k-test $(ts)] ARM $name  exp=$exp  GPU=$gpu  crops=$(basename "$crops")"
    CUDA_VISIBLE_DEVICES=$gpu $PY $VOX/sample.py \
        pretrained_path=$VOX/exps/$exp \
        dset=crossdocked_xray \
        dset.data_dir=$DATA \
        dset.crops_dir=$crops \
        wjs.split=test wjs.start=0 wjs.end=99 wjs.n_targets=100 \
        wjs.n_samples_per_pocket=10 \
        out_dir=res_ep99_test \
        save_dir=$out \
        > $LOG/260519_sample_10k_${name}_test.log 2>&1
    local rc=$?
    echo "[10k-test $(ts)] ARM $name DONE (exit $rc) -> $out"
}

sample_arm density 260518_voxbind_10k_density $DATA/xray_crops       2 &
sample_arm noise   260518_voxbind_10k_noise   $DATA/xray_crops_noise 3 &
wait
echo "[10k-test $(ts)] ALL ARMS DONE"
