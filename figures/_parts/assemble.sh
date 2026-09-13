#!/usr/bin/env bash
# assemble.sh — build figures/draw.py from the pieces in this folder.
#
# draw.py is ONE file on purpose (see its docstring), but it is edited in pieces: the core
# (style, palette, the pose loader, the registry) and one file per figure family. This
# concatenates them in a fixed order -- core, families alphabetically, the runner last --
# so a rebuild is byte-reproducible and two people editing different families never touch
# the same lines.
#
#   bash figures/_parts/assemble.sh
#
# A PART IS NOT A SCRIPT. It is pasted into the middle of draw.py, so a top-level `import`
# or an `if __name__ == "__main__"` block in one does not just look wrong, it CHANGES WHAT
# draw.py DOES -- a stray tool file dropped in this folder once turned `draw.py --list`
# into a different program entirely, silently, because everything still exited 0. So every
# part is checked before anything is written, and a bad one stops the build.
set -euo pipefail
cd "$(dirname "$0")"
# PYTHON overrides the interpreter on a box that does not have this env.

OUT=../draw.py
# A .py suffix, because importlib refuses to load a path it cannot recognise as source.
TMP=../.draw_assembling.py

parts=()
for f in *.py; do
    case "$f" in 00_core.py|zz_cli.py) continue;; esac
    [ -L "$f" ] && continue                 # symlinked tools are not parts
    parts+=("$f")
done

fail=0
for f in "${parts[@]}"; do
    if grep -nE '^(import |from [A-Za-z_])' "$f" | head -3 | grep -q .; then
        echo "ERROR: $f has top-level imports — a part may not import (see CONTRACT.md):" >&2
        grep -nE '^(import |from [A-Za-z_])' "$f" | head -3 | sed 's/^/    /' >&2
        fail=1
    fi
    if grep -qE '^if __name__' "$f"; then
        echo "ERROR: $f has an __main__ block — only zz_cli.py may have one" >&2
        fail=1
    fi
    if ! grep -qE '^@figure\(' "$f"; then
        echo "ERROR: $f registers no figure (@figure) — is it a part at all?" >&2
        fail=1
    fi
done
[ "$fail" -eq 0 ] || { echo "refusing to write $OUT" >&2; exit 1; }

{
    cat 00_core.py
    for f in $(printf '%s\n' "${parts[@]}" | sort); do cat "$f"; done
    cat zz_cli.py
} > "$TMP"

# It must at least import cleanly and register something before it replaces the real file.
n=$("${PYTHON:-/opt/conda/envs/voxbind/bin/python}" - "$TMP" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_probe", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(len(m.FIGURES))
PY
) || { echo "ERROR: assembled file does not import — left at $TMP" >&2; exit 1; }

mv "$TMP" "$OUT"
chmod +x "$OUT"
echo "wrote $(cd .. && pwd)/draw.py  ($(wc -l < "$OUT") lines, ${#parts[@]} families, $n figures)"
