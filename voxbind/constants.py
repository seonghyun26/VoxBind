ATOM_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P", "H", "Br", "I", "B"]
N_POCKET_ELEMENTS = 4
N_LIGAND_ELEMENTS = 7

ELEMENTS_HASH_CROSSDOCKED = {
    "C": 0,
    "O": 1,
    "N": 2,
    "S": 3,
    "F": 4,
    "Cl": 5,
    "P": 6,
    "H": 7,
}

CHANNEL_TO_ATM_NB_CROSSDOCKED = {
    0: 6,   # C
    1: 8,   # O
    2: 7,   # N
    3: 16,  # S
    4: 9,   # F
    5: 17,  # Cl
    6: 15,  # P
    7: 1,   # H
}

# Van-der-Waals radii (Å). The first 11 (C..B) are the original VoxBind set and
# are the ONLY ones indexed by ELEMENTS_HASH_CROSSDOCKED's per-element LUTs — those
# are unchanged. The remainder extend coverage to the diverse elements that appear
# in PLINDER ligands/cofactors/metal sites, so the ROLE-split representation can
# voxelize ANY heavy atom at its own vdW diameter instead of dropping it (see
# vdw_radius() / the Phase-2 diverse build). Sources: Bondi (1964) + Mantina (2009)
# for main-group, Alvarez (2013, Dalton Trans) for transition/heavy metals.
RADIUS_PER_ATOM = {
    "MOL": {
        # ── original VoxBind set (indexed by ELEMENTS_HASH_CROSSDOCKED) ──
        "C": 1.7,
        "O": 1.52,
        "N": 1.55,
        "S": 1.8,
        "F": 1.47,
        "Cl": 1.75,
        "Br": 1.85,
        "P": 1.8,
        "H": 1.2,
        "I": 1.98,
        "B": 1.92,
        # ── diverse main-group (ligand/cofactor) ──
        "Se": 1.90, "Si": 2.10, "As": 1.85, "Te": 2.06, "Sb": 2.06,
        "Sn": 2.17, "Ge": 2.11, "Ga": 1.87, "Al": 1.84, "Pb": 2.02, "Bi": 2.07,
        # ── alkali / alkaline-earth (common pocket ions) ──
        "Li": 1.82, "Na": 2.27, "K": 2.75, "Rb": 3.03, "Cs": 3.43,
        "Mg": 1.73, "Ca": 2.31, "Sr": 2.49, "Ba": 2.68,
        # ── transition / heavy metals (cofactors, catalytic sites) ──
        "V": 2.07, "Cr": 2.06, "Mn": 2.05, "Fe": 2.04, "Co": 2.00, "Ni": 1.97,
        "Cu": 1.96, "Zn": 2.01, "Mo": 2.09, "W": 2.10, "Cd": 2.18, "Hg": 2.23,
        "Pt": 2.13, "Au": 2.14, "Ag": 2.11, "Pd": 2.10, "Ru": 2.13, "Rh": 2.10,
        "Tl": 2.16, "In": 2.13, "Gd": 2.34,
    }
}

# Default vdW radius (Å) for any heavy element absent from the table — a typical
# heavy-atom value so the diverse/role build never has to DROP an atom for lack of
# a radius. (~Bondi C-to-S range; the role channel only needs a plausible blob size.)
DEFAULT_VDW_RADIUS = 1.8


# Heavy elements to EXCLUDE from the diverse role build despite being non-H: noble
# gases (cryo / crystallization probes — Ar/Kr/Xe seen in the PLINDER scan) and
# dummy/unknown placeholders. These are experimental artifacts, not part of the
# biological ligand or pocket, so they must not become role-channel "atoms".
NON_PHYSICAL_ELEMENTS = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "X", "Xx", "Uo", "Q"}


def is_diverse_role_atom(element: str) -> bool:
    """True if `element` should occupy a role channel in the Phase-2 diverse build:
    a real (non-H) atom that is not a crystallization/cryo artifact placeholder."""
    if not element:
        return False
    e = element.strip()
    e = e[0].upper() + e[1:].lower()
    return e not in ("H", "D", "T") and e not in NON_PHYSICAL_ELEMENTS


def vdw_radius(element: str, default: float = DEFAULT_VDW_RADIUS) -> float:
    """vdW radius (Å) for an element symbol, robust to case ('CL'/'cl' → 'Cl').

    Returns `default` for elements not in RADIUS_PER_ATOM (never raises) so the
    Phase-2 diverse PLINDER build can assign a radius to ANY heavy atom — atom
    identity in the role channel is carried by blob SIZE, so a sensible default
    keeps an exotic element in-distribution rather than dropping it.
    """
    if not element:
        return default
    e = element.strip()
    e = e[0].upper() + e[1:].lower()
    return RADIUS_PER_ATOM["MOL"].get(e, default)
