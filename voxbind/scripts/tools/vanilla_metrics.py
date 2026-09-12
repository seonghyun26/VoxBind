"""Write results/task2-drugdesign/VoxBind-vanilla/{metrics.json,SOURCE.txt}.

Called by 86_stage_results_bundle.sh after the sample dir is staged. Reads the
per-target metrics.json the sampler already wrote, so the folder's headline
numbers cannot drift from the molecules beside them.

Two scopes (all 100 test pockets / the 79 density pockets) x two aggregations
(pooled over molecules / unweighted across per-target means, which is what
VoxBind-Ours/metrics.json reports) -- pooling choice moves the dock number by
~0.001 here, but the two folders have to be read side by side.

    python vanilla_metrics.py <results/task2-drugdesign>
"""
import glob
import json
import os
import statistics as st
import sys

SEL = '/home/shpark/prj-denovo/baselines/_shared_data/density_pockets_selected.json'


def ligand_of(target_dir):
    """`<subdir>/<ligbase>.sdf` -- the key density_pockets_selected.json uses."""
    for f in os.listdir(target_dir):
        if f.endswith('.sdf') and '__' in f and f != 'samples.sdf':
            sub, lig = f.split('__', 1)
            return f'{sub}/{lig}'
    return None


def block(rows):
    aggs = [r[1] for r in rows]
    mols = [s for r in rows for s in r[2]]

    def pooled(get):
        v = [get(s) for s in mols]
        v = [x for x in v if x is not None]
        return st.mean(v) if v else None

    def unweighted(key):
        v = [a[key] for a in aggs if a.get(key) is not None]
        return st.mean(v) if v else None

    dock = [a['vina_dock_mean'] for a in aggs if a.get('vina_dock_mean') is not None]
    return {
        'n_targets': len(rows),
        'n_molecules': sum(a['n_total'] for a in aggs),
        'n_valid': sum(a['n_valid'] for a in aggs),
        'vina_pooled_over_molecules': {
            'score': pooled(lambda s: (s.get('vina') or {}).get('score_only')),
            'min': pooled(lambda s: (s.get('vina') or {}).get('minimize')),
            'dock': pooled(lambda s: (s.get('vina') or {}).get('dock')),
        },
        'vina_unweighted_over_targets': {
            'score': unweighted('vina_score_mean'),
            'min': unweighted('vina_min_mean'),
            'dock': unweighted('vina_dock_mean'),
            'dock_per_target_median': st.median(dock) if dock else None,
        },
        'qed_mean': unweighted('qed_mean'),
        'sa_mean': unweighted('sa_mean'),
        'diversity': unweighted('diversity'),
        'validity': unweighted('validity'),
        'high_affinity': unweighted('high_affinity'),
    }


def main(t2):
    root = f'{t2}/VoxBind-vanilla/samples'
    keep = set(json.load(open(SEL))['ligand_filenames'])
    per = []
    for t in sorted(glob.glob(f'{root}/target_*')):
        m = f'{t}/metrics.json'
        if not os.path.exists(m):
            continue
        j = json.load(open(m))
        per.append((ligand_of(t) in keep, j['aggregates'], j.get('samples') or []))
    if not per:
        raise SystemExit(f'no per-target metrics under {root}')

    out = {
        'task': 'drugdesign',
        'method': 'VoxBind vanilla',
        'folder': 'VoxBind-vanilla',
        'benchmark': 'CrossDocked',
        'source_run': 'voxbind/exps/reproduction (svr12)',
        'source_samples': 'voxbind/exps/reproduction/samples/res_test_100',
        'weights': 'exps/exp_sig0.9 checkpoint, EPOCH 86 (verified from the checkpoint\'s own epoch field) -- a local 4-GPU sigma=0.9 training run (2026-02-18 to 2026-03-10), byte-identical to model_zoo/voxbind_sig0.9_crossdocked/checkpoint.pth.tar, which is therefore also epoch 86. NOT epoch 350, and not the paper\'s released weights.',
        'sampled': '2026-06-15 on svr12, 100 molecules per pocket',
        'vina_receptor_scope': 'crop -- the pocket10 box (~400 atoms), NOT the full chain. metrics.json carries no dock_receptor_scope (defaults to crop) and target_*/.vina_cache holds only the pocket10 receptor. The four 260827+ arms and the five baselines are full-receptor, so these vina numbers do not sit in the same column as theirs.',
        'reported_as': 'the only 100-molecules/pocket vanilla sample set on svr12. Read it beside VoxBind-Ours (7,881 records over the same 79 pockets) only with the scope and epoch differences in mind -- Ours is epoch 350 and its vina is full-receptor exhaustiveness 32.',
        'scopes': {'all': block(per), 'density79': block([r for r in per if r[0]])},
        'pose_quality': 'NOT measured for this run: samples[].posecheck (strain) and '
                        'samples[].posebusters are absent -- the eval predates '
                        '73_evaluate_samples.sh gaining them (computed_at 2026-06-15). '
                        'Each molecule does carry an interactions block (n_contacts / '
                        'n_clashes / min_dist / closest pairs).',
        'note': 'Not the _vanilla_ep923/full_eval_ep923 run that '
                'notebook/html/260910/pose_common.py reads on the reporting box -- same '
                'published model, a different sampling run.',
        'report': '../../reports/results_drug_design.html',
    }
    json.dump(out, open(f'{t2}/VoxBind-vanilla/metrics.json', 'w'), indent=2)
    for scope, b in out['scopes'].items():
        print(f"   VoxBind-vanilla {scope:9s} {b['n_targets']:3d} pockets / "
              f"{b['n_molecules']:5d} mols / dock "
              f"{b['vina_unweighted_over_targets']['dock']:.4f}")


if __name__ == '__main__':
    main(sys.argv[1])
