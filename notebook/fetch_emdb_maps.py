"""
CrossDocked → EMDB pipeline (standalone script)
================================================
Runs all 5 pipeline stages with checkpointing, then downloads EMDB maps.

Usage
-----
    conda run -n voxbind python notebook/fetch_emdb_maps.py

Stages
------
  1. Extract unique PDB IDs from CrossDocked .pt files
  2. Batch-resolve PDB IDs → UniProt  (RCSB GraphQL, cached)
  3. UniProt → EM PDB IDs             (RCSB search, cached)
  4. EM PDB IDs → EMDB accessions     (RCSB entry API, cached)
  5. Build crossdocked_pdb_to_emdb.json
  6. Download .map.gz files from EBI FTP

All intermediate results are cached in notebook/data/ so the script
can be interrupted and resumed at any stage.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "voxbind" / "dataset" / "data"
OUT_DIR   = Path(__file__).parent / "data"
EMDB_DIR  = OUT_DIR / "emdb"
EMDB_DIR.mkdir(parents=True, exist_ok=True)

UNIPROT_FILE  = OUT_DIR / "pdb_to_uniprot.json"
EM_PDBS_FILE  = OUT_DIR / "uniprot_to_em_pdbs.json"
EMDB_FILE     = OUT_DIR / "pdb_to_emdb.json"
FINAL_FILE    = OUT_DIR / "crossdocked_pdb_to_emdb.json"

# ── API constants ─────────────────────────────────────────────────────────────
GRAPHQL_URL  = "https://data.rcsb.org/graphql"
RCSB_SEARCH  = "https://search.rcsb.org/rcsbsearch/v2/query"
BATCH_SIZE   = 150
MAX_WORKERS  = 16
SAVE_EVERY   = 200


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Extract unique PDB IDs
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdb_id(pocket_id: str) -> str:
    return pocket_id.split("/")[1].split("_")[0].upper()


def stage1_extract_pdb_ids() -> tuple[list[str], dict[str, list]]:
    print("\n[Stage 1] Extracting PDB IDs from CrossDocked .pt files …")
    pdb_to_pocket_ids: dict[str, list] = {}

    for split_name, pt_file in [("train", "data_train.pt"), ("test", "data_test.pt")]:
        pt_path = DATA_DIR / pt_file
        if not pt_path.exists():
            print(f"  [SKIP] {pt_file} not found")
            continue
        samples = torch.load(pt_path, weights_only=False)
        for pocket_dict, _ in samples:
            pid = extract_pdb_id(pocket_dict["id"])
            pdb_to_pocket_ids.setdefault(pid, []).append(
                (pocket_dict["id"], split_name)
            )
        print(f"  Loaded {pt_file}: {len(samples):,} samples")

    pdb_ids = sorted(pdb_to_pocket_ids)
    print(f"  Unique PDB IDs: {len(pdb_ids):,}")
    return pdb_ids, pdb_to_pocket_ids


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – PDB IDs → UniProt
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_uniprots_batch(
    batch: list[str], session: requests.Session
) -> dict[str, list[str]]:
    ids_str = json.dumps(batch)
    query = f"""
    {{
      entries(entry_ids: {ids_str}) {{
        rcsb_id
        polymer_entities {{
          rcsb_polymer_entity_container_identifiers {{
            reference_sequence_identifiers {{
              database_name
              database_accession
            }}
          }}
        }}
      }}
    }}
    """
    r = session.post(GRAPHQL_URL, json={"query": query}, timeout=30)
    r.raise_for_status()
    result: dict[str, list[str]] = {}
    for entry in r.json().get("data", {}).get("entries") or []:
        pid  = entry["rcsb_id"]
        unis = []
        for entity in entry.get("polymer_entities") or []:
            for ref in (
                entity.get("rcsb_polymer_entity_container_identifiers", {})
                      .get("reference_sequence_identifiers") or []
            ):
                if ref.get("database_name") == "UniProt":
                    unis.append(ref["database_accession"])
        result[pid] = list(dict.fromkeys(unis))
    for pid in batch:
        result.setdefault(pid, [])
    return result


def stage2_pdb_to_uniprot(pdb_ids: list[str]) -> dict[str, list[str]]:
    print("\n[Stage 2] Resolving PDB IDs → UniProt …")
    if UNIPROT_FILE.exists():
        with open(UNIPROT_FILE) as f:
            pdb_to_uniprot: dict[str, list[str]] = json.load(f)
        print(f"  Loaded cache: {len(pdb_to_uniprot):,} entries")
    else:
        pdb_to_uniprot = {}

    to_resolve = [p for p in pdb_ids if p not in pdb_to_uniprot]
    if not to_resolve:
        print("  All cached.")
        return pdb_to_uniprot

    session = requests.Session()
    batches  = [to_resolve[i : i + BATCH_SIZE] for i in range(0, len(to_resolve), BATCH_SIZE)]

    with tqdm(total=len(to_resolve), unit="PDB", desc="  UniProt lookup") as pbar:
        for i, batch in enumerate(batches):
            try:
                result = _fetch_uniprots_batch(batch, session)
                pdb_to_uniprot.update(result)
            except Exception as e:
                tqdm.write(f"  [WARN] batch failed: {e}")
            pbar.update(len(batch))
            if (i + 1) % (SAVE_EVERY // BATCH_SIZE + 1) == 0:
                with open(UNIPROT_FILE, "w") as f:
                    json.dump(pdb_to_uniprot, f)

    with open(UNIPROT_FILE, "w") as f:
        json.dump(pdb_to_uniprot, f, indent=2)
    print(f"  Saved → {UNIPROT_FILE}")
    return pdb_to_uniprot


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – UniProt → EM PDB IDs
# ─────────────────────────────────────────────────────────────────────────────

def _search_em_pdbs(
    uniprot: str, session: requests.Session
) -> tuple[str, list[str]]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers"
                            ".reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": uniprot,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "ELECTRON MICROSCOPY",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"results_verbosity": "compact"},
    }
    try:
        r = session.post(RCSB_SEARCH, json=query, timeout=15)
        if r.status_code == 204:
            return uniprot, []
        r.raise_for_status()
        hits = r.json().get("result_set", [])
        # compact mode returns list of strings
        if hits and isinstance(hits[0], dict):
            hits = [h["identifier"] for h in hits]
        return uniprot, hits
    except Exception:
        return uniprot, []


def stage3_uniprot_to_em_pdbs(
    pdb_to_uniprot: dict[str, list[str]]
) -> dict[str, list[str]]:
    print("\n[Stage 3] Searching EM PDB structures per UniProt …")
    if EM_PDBS_FILE.exists():
        with open(EM_PDBS_FILE) as f:
            uniprot_to_em_pdbs: dict[str, list[str]] = json.load(f)
        print(f"  Loaded cache: {len(uniprot_to_em_pdbs):,} entries")
    else:
        uniprot_to_em_pdbs = {}

    all_uniprots = sorted({u for v in pdb_to_uniprot.values() for u in v})
    to_search    = [u for u in all_uniprots if u not in uniprot_to_em_pdbs]
    if not to_search:
        print("  All cached.")
        return uniprot_to_em_pdbs

    session = requests.Session()
    n_done  = 0

    with tqdm(total=len(to_search), unit="UniProt", desc="  EM PDB search") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_search_em_pdbs, u, session): u for u in to_search}
            for fut in as_completed(futures):
                uni, em_pdbs = fut.result()
                uniprot_to_em_pdbs[uni] = em_pdbs
                n_done += 1
                pbar.update(1)
                if n_done % SAVE_EVERY == 0:
                    with open(EM_PDBS_FILE, "w") as f:
                        json.dump(uniprot_to_em_pdbs, f)

    with open(EM_PDBS_FILE, "w") as f:
        json.dump(uniprot_to_em_pdbs, f, indent=2)
    print(f"  Saved → {EM_PDBS_FILE}")

    uniprots_with_em = {u: v for u, v in uniprot_to_em_pdbs.items() if v}
    print(f"  UniProts with EM structures: {len(uniprots_with_em):,}")
    return uniprot_to_em_pdbs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – EM PDB IDs → EMDB accessions
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_emdb_ids(
    pdb_id: str, session: requests.Session
) -> tuple[str, list[str]]:
    try:
        r = session.get(
            f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=15
        )
        if r.status_code != 200:
            return pdb_id, []
        return pdb_id, [
            rel["db_id"]
            for rel in r.json().get("pdbx_database_related", [])
            if rel.get("db_name") == "EMDB"
        ]
    except Exception:
        return pdb_id, []


def stage4_em_pdb_to_emdb(
    uniprot_to_em_pdbs: dict[str, list[str]]
) -> dict[str, list[str]]:
    print("\n[Stage 4] Resolving EM PDB IDs → EMDB accessions …")
    if EMDB_FILE.exists():
        with open(EMDB_FILE) as f:
            em_pdb_to_emdb: dict[str, list[str]] = json.load(f)
        print(f"  Loaded cache: {len(em_pdb_to_emdb):,} entries")
    else:
        em_pdb_to_emdb = {}

    all_em_pdb_ids = sorted({p for v in uniprot_to_em_pdbs.values() for p in v})
    to_resolve     = [p for p in all_em_pdb_ids if p not in em_pdb_to_emdb]

    if not to_resolve:
        print("  All cached.")
        return em_pdb_to_emdb

    # Sanity-check: warn if cache keys look like CrossDocked PDB IDs
    if em_pdb_to_emdb:
        cached_keys = set(em_pdb_to_emdb.keys())
        overlap = len(cached_keys & set(all_em_pdb_ids))
        if overlap == 0 and len(cached_keys) > 100:
            print(
                "  [WARN] Existing cache has no overlap with EM PDB IDs — "
                "clearing stale cache."
            )
            em_pdb_to_emdb = {}
            to_resolve = all_em_pdb_ids

    session = requests.Session()
    n_done  = 0

    with tqdm(total=len(to_resolve), unit="entry", desc="  EMDB lookup") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_fetch_emdb_ids, p, session): p for p in to_resolve}
            for fut in as_completed(futures):
                pid, emdb_ids = fut.result()
                em_pdb_to_emdb[pid] = emdb_ids
                n_done += 1
                pbar.update(1)
                if n_done % SAVE_EVERY == 0:
                    with open(EMDB_FILE, "w") as f:
                        json.dump(em_pdb_to_emdb, f)

    with open(EMDB_FILE, "w") as f:
        json.dump(em_pdb_to_emdb, f, indent=2)
    print(f"  Saved → {EMDB_FILE}")
    return em_pdb_to_emdb


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – Build final CrossDocked PDB → EMDB mapping
# ─────────────────────────────────────────────────────────────────────────────

def stage5_build_final(
    pdb_ids: list[str],
    pdb_to_uniprot: dict[str, list[str]],
    uniprot_to_em_pdbs: dict[str, list[str]],
    em_pdb_to_emdb: dict[str, list[str]],
) -> dict[str, list[str]]:
    print("\n[Stage 5] Building final CrossDocked → EMDB mapping …")
    crossdocked_to_emdb: dict[str, list[str]] = {}
    for cd_pid in pdb_ids:
        emdb_ids: set[str] = set()
        for uni in pdb_to_uniprot.get(cd_pid, []):
            for em_pid in uniprot_to_em_pdbs.get(uni, []):
                emdb_ids.update(em_pdb_to_emdb.get(em_pid, []))
        crossdocked_to_emdb[cd_pid] = sorted(emdb_ids)

    with open(FINAL_FILE, "w") as f:
        json.dump(crossdocked_to_emdb, f, indent=2)

    with_emdb    = {k: v for k, v in crossdocked_to_emdb.items() if v}
    without_emdb = {k for k, v in crossdocked_to_emdb.items() if not v}
    print(f"  Total CrossDocked PDB IDs : {len(crossdocked_to_emdb):,}")
    print(f"  With EMDB map             : {len(with_emdb):,}")
    print(f"  Without EMDB map          : {len(without_emdb):,}")
    print(f"  Saved → {FINAL_FILE}")
    return crossdocked_to_emdb


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 – Download EMDB maps
# ─────────────────────────────────────────────────────────────────────────────

def emdb_num(emdb_id: str) -> str:
    return emdb_id.split("-")[-1]


def emdb_map_path(emdb_id: str) -> Path:
    return EMDB_DIR / f"emd_{emdb_num(emdb_id)}.map.gz"


def _download_one(emdb_id: str, session: requests.Session) -> tuple[str, bool, int]:
    """Returns (emdb_id, success, bytes_written)."""
    num  = emdb_num(emdb_id)
    dest = emdb_map_path(emdb_id)
    url  = (
        f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
        f"EMD-{num}/map/emd_{num}.map.gz"
    )
    try:
        r = session.get(url, stream=True, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                written += len(chunk)
        return emdb_id, True, written
    except Exception as e:
        if dest.exists():
            dest.unlink()
        return emdb_id, False, 0


def stage6_download_maps(crossdocked_to_emdb: dict[str, list[str]]) -> None:
    print("\n[Stage 6] Downloading EMDB density maps …")
    all_emdb_ids = sorted({e for v in crossdocked_to_emdb.values() for e in v})
    to_download  = [e for e in all_emdb_ids if not emdb_map_path(e).exists()]

    print(f"  Total unique EMDB IDs : {len(all_emdb_ids):,}")
    print(f"  Already downloaded    : {len(all_emdb_ids) - len(to_download):,}")
    print(f"  To download           : {len(to_download):,}")

    if not to_download:
        print("  Nothing to download.")
        return

    session  = requests.Session()
    failed   = []
    total_mb = 0

    with tqdm(
        total=len(to_download),
        unit="map",
        desc="  Downloading",
        dynamic_ncols=True,
    ) as pbar:
        for emdb_id in to_download:
            _, ok, nbytes = _download_one(emdb_id, session)
            mb = nbytes / 1_000_000
            total_mb += mb
            if ok:
                pbar.set_postfix_str(f"{emdb_id}  {mb:.0f} MB  total={total_mb:.0f} MB")
            else:
                failed.append(emdb_id)
                pbar.set_postfix_str(f"{emdb_id} FAILED")
            pbar.update(1)

    print(f"\n  Total downloaded : {total_mb:.0f} MB")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for e in failed:
            print(f"    {e}")
    else:
        print("  All succeeded.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("CrossDocked → EMDB pipeline")
    print("=" * 60)

    pdb_ids, pdb_to_pocket_ids = stage1_extract_pdb_ids()
    pdb_to_uniprot              = stage2_pdb_to_uniprot(pdb_ids)
    uniprot_to_em_pdbs          = stage3_uniprot_to_em_pdbs(pdb_to_uniprot)
    em_pdb_to_emdb              = stage4_em_pdb_to_emdb(uniprot_to_em_pdbs)
    crossdocked_to_emdb         = stage5_build_final(
        pdb_ids, pdb_to_uniprot, uniprot_to_em_pdbs, em_pdb_to_emdb
    )
    stage6_download_maps(crossdocked_to_emdb)

    print("\nDone.")


if __name__ == "__main__":
    main()
