#!/usr/bin/env bash
# Download the cryo-EM datasets used in (or compatible with) the CryoBoltz paper.
#
# CryoBoltz: "Multiscale guidance of AlphaFold3 with heterogeneous cryo-EM data"
#   https://arxiv.org/abs/2506.04490
#   Code: https://github.com/ml-struct-bio/cryoboltz
#
# Paper evaluation targets:
#   Synthetic maps  → STP10 (7AAQ/7AAR) and CH67 Antibody (4HKX) — maps generated
#                     from PDB coords via ChimeraX molmap (not separately hosted)
#   Real EMDB maps  → P-glycoprotein in 4 states (EMDB-40226/40227/40258/40259)
#   README example  → Pma1 monomer (EMDB-64136 / PDB-9UGC)
#
# Usage:
#   bash download_data.sh          # download everything
#   bash download_data.sh pgp      # P-glycoprotein real maps only
#   bash download_data.sh pma1     # Pma1 README example only
#   bash download_data.sh pdb      # PDB structures for synthetic-map targets only

set -euo pipefail
TARGET="${1:-all}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PDB_BASE="https://files.rcsb.org/download"
# EMDB maps are served via EBI FTP (HTTP also works)
EMDB_BASE="https://ftp.ebi.ac.uk/pub/databases/emdb/structures"

download_pdb() {
    local pdb_id="$1" dest="$OUT_DIR/$2"
    mkdir -p "$dest"
    local f="$dest/${pdb_id}.pdb"
    if [ -f "$f" ]; then echo "  [skip] $f already exists"; return; fi
    echo "  Downloading PDB:${pdb_id} → $f"
    curl -fsSL "${PDB_BASE}/${pdb_id}.pdb" -o "$f"
}

download_cif() {
    local pdb_id="$1" dest="$OUT_DIR/$2"
    mkdir -p "$dest"
    local f="$dest/${pdb_id}.cif"
    if [ -f "$f" ]; then echo "  [skip] $f already exists"; return; fi
    echo "  Downloading CIF:${pdb_id} → $f"
    curl -fsSL "${PDB_BASE}/${pdb_id}.cif" -o "$f"
}

download_emdb_map() {
    # EMDB map IDs are zero-padded to 5 digits for old entries,
    # but newer (5-digit) IDs are used as-is.
    local emdb_id="$1" dest="$OUT_DIR/$2"
    mkdir -p "$dest"
    local padded
    padded=$(printf "%05d" "$emdb_id")
    local fname="emd_${padded}.map.gz"
    local f="$dest/$fname"
    if [ -f "${f%.gz}" ]; then echo "  [skip] ${f%.gz} already exists"; return; fi
    echo "  Downloading EMDB-${emdb_id} → $f"
    curl -fsSL "${EMDB_BASE}/EMD-${emdb_id}/map/${fname}" -o "$f"
    echo "  Decompressing $f"
    gunzip -f "$f"
}

echo "=== CryoBoltz dataset download ==="
echo "Output directory: $OUT_DIR"

# ── P-glycoprotein (real EMDB maps) ────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "pgp" ]]; then
    echo ""
    echo "── P-glycoprotein (4 conformational states) ──"
    declare -A PGP_STATES=( ["apo"]="40226 8GMG" ["inward"]="40259 8SA1" ["occluded"]="40258 8SA0" ["collapsed"]="40227 8GMJ" )
    for state in "${!PGP_STATES[@]}"; do
        read -r emdb pdb <<< "${PGP_STATES[$state]}"
        echo " [$state]"
        download_emdb_map "$emdb" "pgp/$state"
        download_pdb      "$pdb"  "pgp/$state"
        download_cif      "$pdb"  "pgp/$state"
    done
fi

# ── Pma1 monomer (README walkthrough example) ──────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "pma1" ]]; then
    echo ""
    echo "── Pma1 monomer (README example) ──"
    download_emdb_map 64136 "pma1"
    download_pdb      9UGC  "pma1"
    download_cif      9UGC  "pma1"
fi

# ── PDB structures for synthetic-map targets ───────────────────────────────────
# (Synthetic maps must be generated via ChimeraX molmap — see inspect.ipynb)
if [[ "$TARGET" == "all" || "$TARGET" == "pdb" ]]; then
    echo ""
    echo "── Synthetic-map target PDB structures ──"
    echo " [STP10 — inward-open and outward-open]"
    download_pdb 7AAQ "synthetic/stp10"
    download_pdb 7AAR "synthetic/stp10"
    download_cif 7AAQ "synthetic/stp10"
    download_cif 7AAR "synthetic/stp10"

    echo " [CH67 Antibody]"
    download_pdb 4HKX "synthetic/ch67"
    download_cif 4HKX "synthetic/ch67"
fi

echo ""
echo "=== Done ==="
echo "Files written to:"
find "$OUT_DIR" -not -path "*/.git/*" -type f | sort
