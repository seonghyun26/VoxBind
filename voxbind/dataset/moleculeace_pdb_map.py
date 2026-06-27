"""Map MoleculeACE benchmark ligands -> PDB co-crystals with electron density.

Feasibility + manifest builder for an EXTERNAL, expert-curated activity-cliff test
with experimental density (vs the self-extracted PDBbind cliffs). For each ChEMBL
target dataset:
  1. ChEMBL target -> UniProt accession.
  2. UniProt -> PDB entry set (PDBe SIFTS best_structures).
  3. Benchmark SMILES -> InChIKey -> matching PDB ligand code(s) (wwPDB CCD inventory).
  4. matched code -> PDB entries containing it, intersect with the target's entries
     (drops same-skeleton ligands bound to OTHER proteins).
  5. keep X-ray entries with structure factors (density obtainable).
  6. within the density-backed subset, find MoleculeACE consensus cliff pairs
     (sim>=0.9 AND |dpKi|>1) -> the usable both-backed cliff pairs.
Outputs voxbind/dataset/data/moleculeace_density/manifest.json + per-target summary.

CPU/network only. Run:  python voxbind/dataset/moleculeace_pdb_map.py
"""
import csv, json, os, sys, types, urllib.request, time
import numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

REPO = "/home/shpark/prj-denovo/VoxBind"
BD = "/tmp/MoleculeACE/MoleculeACE/Data/benchmark_data"
CCD = "/tmp/pdbmap/ccd_inchikey.tsv"
OUT = f"{REPO}/voxbind/dataset/data/moleculeace_density"
os.makedirs(OUT, exist_ok=True)

# levenshtein shim + vendored MoleculeACE cliff code
def _lev(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, 1): cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca != cb))
        prev = cur
    return prev[len(b)]
sys.modules["Levenshtein"] = types.ModuleType("Levenshtein"); sys.modules["Levenshtein"].distance = _lev
sys.path.insert(0, f"{REPO}/notebook"); import moleculeace_cliffs as mace
mace.tqdm = lambda it, **k: it

# targets with the most cliff&inPDB matches (proteases + nuclear receptors + kinases first)
TARGETS = [
    ("CHEMBL244_Ki",  "CHEMBL244"),   # Factor X        (Protease)
    ("CHEMBL204_Ki",  "CHEMBL204"),   # Thrombin        (Protease)
    ("CHEMBL235_EC50","CHEMBL235"),   # PPAR            (NR)
    ("CHEMBL239_EC50","CHEMBL239"),   # PPAR            (NR)
    ("CHEMBL3979_EC50","CHEMBL3979"), # PPAR            (NR)
    ("CHEMBL2047_EC50","CHEMBL2047"), # Farnesoid X rec (NR)
    ("CHEMBL2034_Ki", "CHEMBL2034"),  # Glucocorticoid  (NR)
    ("CHEMBL1871_Ki", "CHEMBL1871"),  # Androgen rec    (NR)
    ("CHEMBL2835_Ki", "CHEMBL2835"),  # JAK1            (Kinase)
    ("CHEMBL2971_Ki", "CHEMBL2971"),  # JAK2            (Kinase)
    ("CHEMBL262_Ki",  "CHEMBL262"),   # GSK-3           (Kinase)
    ("CHEMBL4005_Ki", "CHEMBL4005"),  # PI3K alpha      (Transferase)
]

def get(url):
    for _ in range(2):
        try:
            with urllib.request.urlopen(url, timeout=25) as r: return json.load(r)
        except Exception: time.sleep(0.4)
    return {}

# CCD InChIKey inventory
blk2codes, full2codes = {}, {}
for line in open(CCD):
    code, ik = line.rstrip().split("\t")
    blk2codes.setdefault(ik.split("-")[0], set()).add(code)
    full2codes.setdefault(ik, set()).add(code)
print(f"CCD inventory: {len(full2codes)} full keys / {len(blk2codes)} blocks", flush=True)

manifest, summary = {}, []
for ds, chembl in TARGETS:
    t = get(f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl}.json")
    accs = [c.get("accession") for c in t.get("target_components", []) if c.get("accession")]
    pdb_set = set()
    for acc in accs:
        bs = get(f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{acc}")
        pdb_set |= {e["pdb_id"].lower() for e in bs.get(acc, [])}
    # benchmark molecules matched to a CCD code
    mols = []
    for r in csv.DictReader(open(f"{BD}/{ds}.csv")):
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None: continue
        ik = Chem.MolToInchiKey(m)
        codes = full2codes.get(ik, set()) or blk2codes.get(ik.split("-")[0], set())
        if codes:
            mols.append(dict(smiles=r["smiles"], ik=ik, codes=sorted(codes),
                             cliff=r["cliff_mol"] in ("1", "True", "true"),
                             pKi=float(r["y [pEC50/pKi]"]), split=r["split"]))
    # resolve which matched codes are bound to THIS target + has structure factors
    code2ent = {}
    for c in sorted({c for m in mols for c in m["codes"]}):
        d = get(f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/in_pdb/{c}")
        inter = {p["pdb_id"].lower() for p in d.get(c, [])} & pdb_set
        if inter: code2ent[c] = sorted(inter)
    cand_ent = sorted({p for v in code2ent.values() for p in v})
    sf_res = {}
    for p in cand_ent:
        ex = get(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/experiment/{p}").get(p, [])
        is_xray = any(str(r.get("experimental_method", "")).lower().startswith("x-ray") for r in ex)
        res = next((r.get("resolution") for r in ex if r.get("resolution")), None)
        if is_xray: sf_res[p] = res
    # density-backed molecules: pick best (highest-res) target entry
    dens = []
    for m in mols:
        ents = sorted({p for c in m["codes"] if c in code2ent for p in code2ent[c] if p in sf_res})
        if ents:
            ents.sort(key=lambda p: (sf_res[p] if sf_res[p] is not None else 9.9))
            m2 = dict(m); m2["entries"] = ents; m2["best"] = ents[0]; m2["best_res"] = sf_res[ents[0]]
            dens.append(m2)
    # cliff pairs within density-backed subset (fast: small n)
    pairs = []
    if len(dens) >= 2:
        S = mace.moleculeace_similarity([m["smiles"] for m in dens], 0.9, hide=True)
        pairs = [(i, j) for i in range(len(dens)) for j in range(i+1, len(dens))
                 if S[i][j] == 1 and abs(dens[i]["pKi"] - dens[j]["pKi"]) > 1.0]
    pair_mols = sorted({i for p in pairs for i in p})
    manifest[ds] = dict(chembl=chembl, uniprot=accs, n_pdb_entries=len(pdb_set),
                        n_matched=len(mols), n_density=len(dens),
                        density_mols=dens, pairs=pairs)
    summary.append(dict(ds=ds, chembl=chembl, uniprot=accs[0] if accs else None,
                        n_pdb=len(pdb_set), matched=len(mols), density=len(dens),
                        density_cliff=sum(m["cliff"] for m in dens),
                        pairs=len(pairs), pair_mols=len(pair_mols)))
    s = summary[-1]
    print(f"{ds:16} {chembl:11} entries={s['n_pdb']:4} matched={s['matched']:4} "
          f"density={s['density']:4} pairs={s['pairs']:3} pairmols={s['pair_mols']:3}", flush=True)

json.dump(manifest, open(f"{OUT}/manifest.json", "w"))
json.dump(summary, open(f"{OUT}/summary.json", "w"))
tp = sum(s["pairs"] for s in summary); tm = sum(s["pair_mols"] for s in summary); td = sum(s["density"] for s in summary)
print(f"\nTOTAL: density-backed mols={td}  both-backed cliff pairs={tp}  distinct pair-mols={tm}")
print("saved ->", f"{OUT}/manifest.json")
