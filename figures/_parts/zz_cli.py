

# ════════════════════════════════════════════════════════════════════════════════
# the runner
# ════════════════════════════════════════════════════════════════════════════════
def _by_eval():
    """{eval: [fig_id, ...]} in registration order — `fig-<eval>-<name>` splits on the
    SECOND hyphen, so a figure name may itself contain hyphens."""
    out = collections.OrderedDict()
    for fid in FIGURES:
        ev = fid.split("-")[1]
        out.setdefault(ev, []).append(fid)
    return out


def _list():
    print(f"{len(FIGURES)} figures in {len(_by_eval())} evaluations "
          f"-> {OUT_ROOT.relative_to(REPO)}/fig-<eval>/\n")
    for ev, ids in _by_eval().items():
        folders = sorted({figure_folder(f) for f in ids})
        print(f"  {ev}  ->  {', '.join(f + '/' for f in folders)}")
        for fid in ids:
            doc = (FIGURES[fid].__doc__ or "").strip().splitlines()
            here = figure_folder(fid)
            tag = f"[{here.split('/')[-1]}] " if len(folders) > 1 else ""
            print(f"    {fid:44s} {tag}{doc[0] if doc else ''}")
        print()


def _resolve(names):
    """Figure ids for the names asked for. A name that is an EVALUATION expands to every
    figure in it, so `draw.py posecheck` is the whole PoseCheck set. An unknown name is a
    hard error listing the near misses -- silently drawing nothing is the failure mode
    this exists to prevent."""
    by_eval = _by_eval()
    out = []
    for n in names:
        if n in FIGURES:
            out.append(n)
        elif n in by_eval:
            out.extend(by_eval[n])
        else:
            near = [f for f in FIGURES if n in f] or [f"{e} (whole evaluation)"
                                                      for e in by_eval if n in e]
            raise SystemExit(f"unknown figure {n!r}"
                             + (f"\n  did you mean: {', '.join(near)}" if near else
                                f"\n  --list shows all {len(FIGURES)}"))
    seen, uniq = set(), []
    for f in out:                       # keep order, drop repeats
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="draw.py", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A name may be a figure id or a whole evaluation "
               "(vina, posebusters, posecheck, consistency, interaction, molweight, "
               "similarity, mcp, ensemble).")
    ap.add_argument("names", nargs="*", help="figure ids and/or evaluation names")
    ap.add_argument("-a", "--all", action="store_true", help="draw every figure")
    ap.add_argument("-l", "--list", action="store_true", help="list the figures and exit")
    ap.add_argument("-f", "--formats", default="png,svg,pdf",
                    help="comma-separated: png,svg,pdf (default all three)")
    ap.add_argument("-o", "--out", default=str(OUT_ROOT),
                    help="output root; each figure lands in <root>/fig-<eval>/ "
                         f"(default {OUT_ROOT})")
    ap.add_argument("-k", "--keep-going", action="store_true",
                    help="carry on after a figure fails, and report at the end")
    args = ap.parse_args(argv)

    if args.list or (not args.names and not args.all):
        _list()
        return 0

    global _FORMATS
    _FORMATS = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    bad = [f for f in _FORMATS if f not in ("png", "svg", "pdf")]
    if bad:
        raise SystemExit(f"unsupported format(s): {', '.join(bad)}")

    ids = list(FIGURES) if args.all else _resolve(args.names)
    root = Path(args.out)
    use_style()
    print(f"drawing {len(ids)} figure(s) as {'+'.join(_FORMATS)} -> {root}/\n")

    failed, t_all = [], time.time()
    for i, fid in enumerate(ids, 1):
        # The folder is the EVALUATION, not the figure: everything answering the same
        # question sits together, and the shared CSVs two figures both export (the pose
        # per-atom table, the interaction table) land once instead of once per figure.
        out = root / figure_folder(fid)
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(ids)}] {fid}  -> {out.name}/")
        t0 = time.time()
        try:
            FIGURES[fid](out)
        except Exception as exc:                                  # noqa: BLE001
            if not args.keep_going:
                raise
            failed.append((fid, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED {type(exc).__name__}: {exc}")
            continue
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\n{len(ids) - len(failed)}/{len(ids)} drawn in {time.time() - t_all:.1f}s")
    if failed:
        print(f"{len(failed)} failed:")
        for fid, why in failed:
            print(f"  {fid}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
