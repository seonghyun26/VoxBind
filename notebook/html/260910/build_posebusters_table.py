"""build_posebusters_table.py — PoseBusters (+ the PoseCheck numbers beside it)
for our sampled runs and the five published baselines, in one table.

PoseBusters is the validity layer PoseCheck does not cover: 20 graded checks per
molecule (bond lengths/angles, ring & double-bond flatness, internal clash,
internal energy, and the protein-side distance / volume-overlap tests), with
`PB-valid` = every one of them passing. It is reported as a percentage of
molecules, pooled over molecules exactly as the baseline PoseCheck table pools.

Both sides go through the SAME scorer — notebook/webapp/pose_eval.py,
PoseBusters(config="dock") against the WHOLE receptor from
targetdiff/data/test_set, unprotonated — so the column is comparable across
rows. (The PoseCheck columns are NOT equally comparable: the baselines were
protonated with pdb2pqr in baselines/_scripts/run_posecheck.py while our runs go
through posecheck's own hydride/reduce path. They are shown for context, not as
a head-to-head.)

Two pocket scopes are written, because the rows do not all cover the same set:
  * all      — every pocket a row has (100 for the baselines and for our
               260827 base run, 79 for the fusion runs)
  * density79 — the 79 test pockets with usable deposited electron density,
               the subset tables 1b/2b of baseline.html use. The only
               apples-to-apples scope.

    python notebook/html/260910/build_posebusters_table.py
"""
import csv
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = '/home/shpark/prj-denovo/Voxbind/voxbind'
BASE = '/home/shpark/prj-denovo/baselines'
SEL = json.load(open(f'{BASE}/_shared_data/density_pockets_selected.json'))
KEEP = set(SEL['ligand_filenames'])

# Informational columns — "loaded ok", not a geometric check. Same exclusion
# pose_eval.py applies when it sets `valid`.
INFO_COLS = frozenset({'mol_pred_loaded', 'mol_cond_loaded'})

OURS = [
    ('VoxBind base (ours, ep350)',   f'{VOX}/exps/260827_voxbind_base_8gpu/samples/samples_ep350_test'),
    ('Fusion v4 cdgv2 warm ep100',   f'{VOX}/exps/260831_fusion_v4_cdgv2_8gpu/samples/samples_ep100_test79'),
    ('Fusion v4 cdgv2 scratch ep350', f'{VOX}/exps/260902_fusion_v4_cdgv2_scratch_8gpu/samples/samples_ep350_test79'),
    ('Fusion v4 cv2 scratch ep350',  f'{VOX}/exps/260905_fusion_v4_cv2_scratch_8gpu/samples/samples_ep350_test79'),
]
BASELINES = ['AR', 'Pocket2Mol', 'DiffSBDD', 'DecompDiff_ref_prior', 'FuncBind']


def _blank():
    return {'n_mols': 0, 'n_pb': 0, 'n_pb_valid': 0, 'n_pb_err': 0,
            'n_pb_missing': 0,
            'check_pass': {}, 'check_seen': {}, 'pockets': set(),
            'clashes': [], 'strain': [], 'vina_dock': [], 'vina_score': []}


def _add_pb(acc, rec):
    # No block at all means "not scored yet" — a run still in flight, not a
    # failure. Keeping the two apart is what makes a partial table readable.
    if rec is None:
        acc['n_pb_missing'] += 1
        return
    if not isinstance(rec, dict) or 'error' in rec:
        acc['n_pb_err'] += 1
        return
    acc['n_pb'] += 1
    acc['n_pb_valid'] += bool(rec.get('valid'))
    for k, v in (rec.get('checks') or {}).items():
        if k in INFO_COLS or v is None:
            continue
        acc['check_seen'][k] = acc['check_seen'].get(k, 0) + 1
        acc['check_pass'][k] = acc['check_pass'].get(k, 0) + bool(v)


def _finish(label, acc):
    row = {
        'row': label,
        'n_pockets': len(acc['pockets']),
        'n_molecules': acc['n_mols'],
        'pb_scored': acc['n_pb'],
        'pb_errors': acc['n_pb_err'],
        'pb_not_scored': acc['n_pb_missing'],
        'pb_valid_pct': 100.0 * acc['n_pb_valid'] / acc['n_pb'] if acc['n_pb'] else None,
        'checks_pass_rate': {k: acc['check_pass'][k] / acc['check_seen'][k]
                             for k in sorted(acc['check_seen'])},
    }
    for key, vals, how in (('clash_mean', acc['clashes'], 'mean'),
                           ('clash_median', acc['clashes'], 'median'),
                           ('strain_median', acc['strain'], 'median'),
                           ('vina_score_mean', acc['vina_score'], 'mean'),
                           ('vina_dock_mean', acc['vina_dock'], 'mean')):
        row[key] = (round(st.mean(vals), 4) if how == 'mean' else
                    round(st.median(vals), 4)) if vals else None
    return row


def target_ligand_filename(target_dir):
    """`<subdir>/<ligbase>.sdf` for a target dir — the key the baselines use.

    Our sample dirs name the reference ligand `<subdir>__<ligbase>.sdf`, the
    same two halves joined by a double underscore.
    """
    for f in os.listdir(target_dir):
        if f.endswith('.sdf') and '__' in f and f != 'samples.sdf':
            sub, lig = f.split('__', 1)
            return f'{sub}/{lig}'
    return None


def collect_ours(label, sample_dir, keep=None):
    acc = _blank()
    for mj in sorted(glob.glob(f'{sample_dir}/target_*/metrics.json')):
        tdir = os.path.dirname(mj)
        lig = target_ligand_filename(tdir)
        if keep is not None and lig not in keep:
            continue
        acc['pockets'].add(tdir)
        d = json.load(open(mj))
        for s in d.get('samples', []):
            acc['n_mols'] += 1
            _add_pb(acc, s.get('posebusters'))
            pc = s.get('posecheck')
            if isinstance(pc, dict) and 'error' not in pc:
                if isinstance(pc.get('clashes'), (int, float)):
                    acc['clashes'].append(pc['clashes'])
                if isinstance(pc.get('strain'), (int, float)):
                    acc['strain'].append(pc['strain'])
            v = s.get('vina')
            if isinstance(v, dict):
                if isinstance(v.get('dock'), (int, float)):
                    acc['vina_dock'].append(v['dock'])
                if isinstance(v.get('score_only'), (int, float)):
                    acc['vina_score'].append(v['score_only'])
    return _finish(label, acc)


def collect_baseline(name, keep=None):
    acc = _blank()
    for f in sorted(glob.glob(f'{BASE}/_posebusters/{name}/pb_*.json')):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001 — a chunk still being written
            continue
        if keep is not None and d['ligand_filename'] not in keep:
            continue
        acc['pockets'].add(d['pocket_index'])
        for rec in d['posebusters']:
            acc['n_mols'] += 1
            _add_pb(acc, rec)
    row = _finish(name, acc)
    # PoseCheck for the baselines is already summarised; carry its clash/strain
    # across rather than re-reading 2.4k chunk files.
    pc_file = ('posecheck_summary_density79.json' if keep is not None
               else 'posecheck_summary.json')
    try:
        for r in json.load(open(f'{BASE}/_eval/{pc_file}')):
            if r['baseline'] == name:
                row['clash_mean'] = r.get('clash_mean')
                row['clash_median'] = r.get('clash_median')
                row['strain_median'] = r.get('strain_median')
    except FileNotFoundError:
        pass
    # ...and the Vina numbers from the same eval pass.
    vina_file = ('summary_density79.json' if keep is not None else 'summary.json')
    try:
        s = json.load(open(f'{BASE}/_eval/{vina_file}'))
        rows = s if isinstance(s, list) else s.get('rows', [])
        for r in rows:
            if r.get('baseline') == name or r.get('name') == name:
                for src, dst in (('vina_score_mean', 'vina_score_mean'),
                                 ('vina_dock_mean', 'vina_dock_mean')):
                    if r.get(src) is not None:
                        row[dst] = r[src]
    except (FileNotFoundError, AttributeError, TypeError):
        pass
    return row


def build(scope):
    keep = KEEP if scope == 'density79' else None
    rows = [collect_ours(lbl, d, keep) for lbl, d in OURS if os.path.isdir(d)]
    rows += [collect_baseline(b, keep) for b in BASELINES
             if os.path.isdir(f'{BASE}/_posebusters/{b}')]
    return rows


def fmt(rows, scope):
    out = [f'## PoseBusters — scope: {scope}', '',
           '| row | pockets | mols | PB-valid % | PB scored | PB err | '
           'not scored | clash mean/med | strain med | vina score | vina dock |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def n(v, p=2):
        return '—' if v is None else f'{v:.{p}f}'
    for r in rows:
        out.append(
            f"| {r['row']} | {r['n_pockets']} | {r['n_molecules']} | "
            f"{n(r['pb_valid_pct'], 1)} | {r['pb_scored']} | "
            f"{r['pb_errors']} | {r['pb_not_scored']} | "
            f"{n(r['clash_mean'])}/{n(r['clash_median'], 1)} | "
            f"{n(r['strain_median'], 1)} | {n(r['vina_score_mean'])} | "
            f"{n(r['vina_dock_mean'])} |")
    return '\n'.join(out)


if __name__ == '__main__':
    os.makedirs(HERE, exist_ok=True)
    allrows = {}
    for scope in ('all', 'density79'):
        rows = build(scope)
        allrows[scope] = rows
        print(fmt(rows, scope), '\n')
        with open(f'{HERE}/posebusters_table_{scope}.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['row', 'n_pockets', 'n_molecules', 'pb_scored', 'pb_errors',
                        'pb_not_scored', 'pb_valid_pct', 'clash_mean', 'clash_median', 'strain_median',
                        'vina_score_mean', 'vina_dock_mean'])
            for r in rows:
                w.writerow([r['row'], r['n_pockets'], r['n_molecules'], r['pb_scored'],
                            r['pb_errors'], r['pb_not_scored'], r['pb_valid_pct'], r['clash_mean'],
                            r['clash_median'], r['strain_median'],
                            r['vina_score_mean'], r['vina_dock_mean']])
    with open(f'{HERE}/posebusters_table.json', 'w') as fh:
        json.dump(allrows, fh, indent=2)
    print(f'wrote {HERE}/posebusters_table.json and posebusters_table_*.csv')
