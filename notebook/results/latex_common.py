"""latex_common.py — shared HTML→LaTeX machinery for the latex-task{1,2,3} notebooks.

Extracted from results2latex.ipynb (2026-09-08 split). Generic helpers + the generic
`table_to_latex` renderer live here; each notebook imports `*` and keeps its own
task-specific converters. Run notebooks from anywhere — resolve_html_path searches up-tree.
"""
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

LINEBREAK = "\ue000"
INDENT = "    "   # one indent level in the emitted LaTeX; set to "\t" for hard tabs


def resolve_html_path(path: Path) -> Path:
    candidates = [prefix / path for prefix in (Path("."), Path(".."), Path("../.."), Path("../../.."))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"results.html을 찾지 못했습니다. 확인한 경로: {tried}")


def escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"$<$",
        ">": r"$>$",
    }
    text = "".join(replacements.get(char, char) for char in text)
    unicode_replacements = {
        "\xa0": " ",
        "±": r"$\pm$",
        "ρ": r"$\rho$",
        "σ": r"$\sigma$",
        "Δ": r"$\Delta$",
        "↓": r"$\downarrow$",
        "↑": r"$\uparrow$",
        "→": r"$\rightarrow$",
        "←": r"$\leftarrow$",
        "≤": r"$\leq$",
        "≥": r"$\geq$",
        "≈": r"$\approx$",
        "×": r"$\times$",
        "·": r"$\cdot$",
        "−": "-",
        "—": "---",
        "–": "--",
        "²": r"\textsuperscript{2}",
        "†": r"\textsuperscript{$\dagger$}",
        "\u200a": " ",
    }
    for old, new in unicode_replacements.items():
        text = text.replace(old, new)
    return text


def inline_to_latex(node, keep_badges: bool = True) -> str:
    if isinstance(node, NavigableString):
        return escape_latex_text(str(node))
    if not isinstance(node, Tag) or node.name in {"script", "style"}:
        return ""
    if node.name == "br":
        return LINEBREAK

    content = "".join(inline_to_latex(child, keep_badges) for child in node.children)
    classes = set(node.get("class", []))

    if "tag" in classes:
        if not keep_badges or not content.strip():
            return ""
        return rf"\,\textsuperscript{{\scriptsize {content.strip()}}}"
    if "nsub" in classes:
        return LINEBREAK + content
    if node.name in {"b", "strong"}:
        return rf"\textbf{{{content}}}"
    if node.name in {"i", "em"}:
        return rf"\textit{{{content}}}"
    if node.name == "sub":
        # class="sc" 는 \scriptsize 를 덧붙입니다 — de novo 표의 sigma 표기와 같은 모양.
        size = r"\scriptsize " if "sc" in classes else ""
        return rf"\textsubscript{{{size}{content}}}"
    if node.name == "sup":
        return rf"\textsuperscript{{{content}}}"
    if "display:block" in node.get("style", "").replace(" ", "").lower():
        return LINEBREAK + content
    return content


def cell_to_latex(
    cell: Tag, keep_badges: bool = True, preserve_emphasis: bool = True
) -> str:
    raw_parts = inline_to_latex(cell, keep_badges).split(LINEBREAK)
    parts = [re.sub(r"\s+", " ", part).strip() for part in raw_parts]
    parts = [part for part in parts if part]
    if not parts:
        return ""

    classes = set(cell.get("class", []))
    style = cell.get("style", "").replace(" ", "").lower()
    is_bold = cell.name == "th" or "best" in classes or bool(re.search(r"font-weight:(?:[7-9]00|bold)", style))
    is_underlined = "second" in classes or "text-decoration:underline" in style
    if preserve_emphasis and is_bold:
        parts = [rf"\textbf{{{part}}}" for part in parts]
    if preserve_emphasis and is_underlined:
        parts = [rf"\underline{{{part}}}" for part in parts]

    if len(parts) == 1:
        return parts[0]
    align = "c" if cell.name == "th" or "metric" in classes else "l"
    return rf"\shortstack[{align}]{{" + r" \\ ".join(parts) + "}"


@dataclass(frozen=True)
class CellPlacement:
    node: Tag
    row: int
    col: int
    rowspan: int
    colspan: int


def build_layout(table: Tag):
    rows = table.find_all("tr")
    occupied = {}
    placements = []
    ncols = 0

    for row_index, row in enumerate(rows):
        col_index = 0
        for cell in row.find_all(["th", "td"], recursive=False):
            while (row_index, col_index) in occupied:
                col_index += 1
            rowspan = max(1, int(cell.get("rowspan", 1)))
            colspan = max(1, int(cell.get("colspan", 1)))
            placement = CellPlacement(cell, row_index, col_index, rowspan, colspan)
            placements.append(placement)
            for rr in range(row_index, row_index + rowspan):
                for cc in range(col_index, col_index + colspan):
                    occupied[(rr, cc)] = placement
            col_index += colspan
        ncols = max(ncols, col_index)
    return rows, occupied, placements, ncols


def table_title(table: Tag, index: int) -> str:
    title_node = table.find_previous("p", class_="table-title")
    if title_node is not None:
        return re.sub(r"\s+", " ", title_node.get_text(" ", strip=True))
    heading = table.find_previous(["h1", "h2", "h3", "h4"])
    return heading.get_text(" ", strip=True) if heading else f"Results table {index + 1}"


def caption_from_title(title: str) -> str:
    caption = re.sub(r"^Table\s+[A-Za-z0-9.]+\s*[·:—-]\s*", "", title, flags=re.IGNORECASE)
    return escape_latex_text(caption)


def label_from_title(title: str, index: int) -> str:
    match = re.match(r"^Table\s+([A-Za-z0-9.]+)", title, flags=re.IGNORECASE)
    stem = match.group(1) if match else str(index + 1)
    stem = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return f"tab:results-{stem}"


def infer_alignments(placements, ncols: int) -> str:
    alignments = []
    body_cells = [p for p in placements if p.node.name == "td"]
    for col in range(ncols):
        candidates = [p.node for p in body_cells if p.col <= col < p.col + p.colspan]
        cell = candidates[0] if candidates else None
        if cell is None:
            alignments.append("c")
            continue
        classes = set(cell.get("class", []))
        style = cell.get("style", "").replace(" ", "").lower()
        if "text-align:right" in style:
            alignments.append("r")
        elif "metric" in classes or "text-align:center" in style:
            alignments.append("c")
        else:
            alignments.append("l")
    return "".join(alignments)


def parse_mean_std(cell: Tag):
    value_node = cell.find(class_="val")
    if value_node is None:
        return None
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    value_match = re.search(number, value_node.get_text(strip=True).replace("−", "-"))
    if value_match is None:
        return None
    sd_node = cell.find(class_="sd")
    sd_match = re.search(number, sd_node.get_text(strip=True)) if sd_node else None
    mean = float(value_match.group())
    std = abs(float(sd_match.group())) if sd_match else 0.0
    return mean, std


def apply_std_macro(value: str) -> str:
    """Replace an inline standard deviation with the LaTeX std macro."""
    pattern = r"\s*\$\\pm\$\s*([0-9]+(?:\.[0-9]+)?)"
    return re.sub(pattern, lambda match: rf"\std{{{match.group(1)}}}", value)


def method_name_to_latex(cell: Tag) -> str:
    """Keep only the method name, dropping badges and detail/note spans."""
    parts = []
    for child in cell.children:
        if isinstance(child, NavigableString):
            parts.append(escape_latex_text(str(child)))
        elif isinstance(child, Tag) and child.name in {"b", "strong", "i", "em"}:
            parts.append(inline_to_latex(child, keep_badges=False))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def format_two_decimal_places(value: str) -> str:
    """Use two decimal places, padding missing precision invisibly."""
    value = value.strip()
    match = re.fullmatch(r"([+-]?\d+)(?:\.(\d+))?", value)
    if match is None:
        return value
    decimals = match.group(2) or ""
    if len(decimals) > 2:
        rounded = Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return f"{rounded:.2f}"
    if len(decimals) == 2:
        return value
    if len(decimals) == 1:
        return value + r"\phantom{0}"
    return value + r".\phantom{0}\phantom{0}"


def format_pair_two_decimal_places(value: str) -> str:
    return " / ".join(format_two_decimal_places(part) for part in value.split(" / "))


def format_similarity_two_decimal_places(value: str) -> str:
    if " / " in value:
        return format_pair_two_decimal_places(value)
    if value.endswith(r"\%"):
        return format_two_decimal_places(value[:-2]) + r"\%"
    return format_two_decimal_places(value)


def table1_metric_emphasis(rows, placements, metric_columns):
    """Rank cells, requiring mutual mean inclusion for the bold tier."""
    values_by_column = {col: [] for col in metric_columns}
    for placement in placements:
        if placement.col not in values_by_column or placement.node.name != "td":
            continue
        row = rows[placement.row]
        if row.select_one(".tag.leaked, .cat-leaked") is not None:
            continue
        parsed = parse_mean_std(placement.node)
        if parsed is not None:
            values_by_column[placement.col].append((placement.row, *parsed))

    emphasis = {}
    for col, values in values_by_column.items():
        if not values:
            continue
        maximize = (col - 2) % 3 != 2  # Pearson/Spearman ↑, RMSE ↓
        best_mean = (max if maximize else min)(mean for _, mean, _ in values)
        best_std = max(std for _, mean, std in values if abs(mean - best_mean) < 1e-12)
        sota_rows = set()
        for row, mean, std in values:
            mean_gap = abs(mean - best_mean)
            if mean_gap <= best_std and mean_gap <= std:
                emphasis[(row, col)] = "bold"
                sota_rows.add(row)

        remaining = [(row, mean) for row, mean, _ in values if row not in sota_rows]
        if remaining:
            second_mean = (max if maximize else min)(mean for _, mean in remaining)
            for row, mean in remaining:
                if abs(mean - second_mean) < 1e-12:
                    emphasis[(row, col)] = "underline"
    return emphasis

def table_to_latex(table: Tag, index: int, keep_badges: bool = True, resize_wide: bool = True) -> str:
    # (task-specific dispatch removed — each latex-taskN notebook calls its own converter)
    rows, occupied, placements, ncols = build_layout(table)
    title = table_title(table, index)
    caption = (
        TABLE1_CAPTION if index == 0
        else SIMILARITY_CAPTION if index == 1
        else caption_from_title(title)
    )
    label = label_from_title(title, index)
    all_alignments = infer_alignments(placements, ncols)
    visible_columns = list(range(ncols))
    custom_headers = {}
    if index == 0:
        custom_headers = dict(TABLE1_GROUP_HEADERS)
        visible_columns = [0, 1] + [
            col
            for start_col, _ in TABLE1_GROUP_HEADERS
            for col in range(start_col, start_col + 3)
        ]
    elif index == 1:
        custom_headers = dict(SIMILARITY_GROUP_HEADERS)
        visible_columns = [0, 1] + [
            col
            for start_col, _ in SIMILARITY_GROUP_HEADERS
            for col in range(start_col, start_col + 3)
        ]
    metric_emphasis = (
        table1_metric_emphasis(rows, placements, set(visible_columns) - {0, 1})
        if index in {0, 1} else {}
    )
    alignments = "".join(all_alignments[col] for col in visible_columns)
    wide = ncols >= 9
    environment = "table*" if wide else "table"

    header_rows = {
        i for i, row in enumerate(rows)
        if row.find_parent("thead") is not None
    }
    if not header_rows and rows and rows[0].find_all("th", recursive=False):
        header_rows = {0}
    last_header_row = max(header_rows) if header_rows else None

    latex_rows = []
    for row_index in range(len(rows)):
        if (
            index in {0, 1}
            and last_header_row is not None
            and row_index > last_header_row + 1
            and rows[row_index].find("td", class_="col-modality", recursive=False)
            is not None
        ):
            latex_rows.append(r"\midrule")
        tokens = []
        visible_index = 0
        while visible_index < len(visible_columns):
            col_index = visible_columns[visible_index]
            placement = occupied.get((row_index, col_index))
            if placement is None:
                tokens.append("")
                visible_index += 1
                continue
            visible_span = [
                col for col in visible_columns
                if placement.col <= col < placement.col + placement.colspan
            ]

            if placement.row == row_index and col_index == visible_span[0]:
                if index in {0, 1} and row_index == 0 and placement.col in custom_headers:
                    header = escape_latex_text(custom_headers[placement.col])
                    value = rf"\textbf{{{header}}}"
                else:
                    if index == 5 and placement.col == 0 and placement.node.name == "td":
                        value = method_name_to_latex(placement.node)
                    else:
                        value = cell_to_latex(
                            placement.node,
                            False
                            if (index == 0 and TABLE1_EXCLUDE_TAGS)
                            or (index == 1 and SIMILARITY_EXCLUDE_TAGS)
                            or (index == 5 and REFERENCE_SIMILARITY_EXCLUDE_TAGS)
                            else keep_badges,
                            preserve_emphasis=(index not in {0, 1} or placement.node.name == "th"),
                        )
                    if index in {0, 1}:
                        value = apply_std_macro(value)
                    emphasis = metric_emphasis.get((row_index, placement.col))
                    if emphasis == "bold":
                        value = rf"\textbf{{{value}}}"
                    elif emphasis == "underline":
                        value = rf"\underline{{{value}}}"
                if placement.rowspan > 1:
                    value = rf"\multirow{{{placement.rowspan}}}{{*}}{{{value}}}"
                visible_colspan = len(visible_span)
                if visible_colspan > 1:
                    value = rf"\multicolumn{{{visible_colspan}}}{{c}}{{{value}}}"
                tokens.append(value)
                visible_index += visible_colspan
            else:
                # 이전 행에서 시작한 multirow가 차지하는 열의 자리표시자입니다.
                span = len(visible_span) if col_index == visible_span[0] else 1
                tokens.append(rf"\multicolumn{{{span}}}{{c}}{{}}" if span > 1 else "")
                visible_index += span

        latex_rows.append(" & ".join(tokens).lstrip() + r" \\ ".rstrip())
        if row_index == last_header_row:
            latex_rows.append(r"\midrule")

    if index in {0, 1}:
        compact_label = (
            "tab:results-binding-affinity"
            if index == 0 else "tab:results-binding-affinity-similarity"
        )
        body = [
            r"\begin{table}[!t]",
            r"    \centering",
            r"    \caption{",
            f"        {caption}",
            r"    }",
            rf"    \label{{{compact_label}}}",
            r"    \resizebox{.98\textwidth}{!}{%",
            rf"        \begin{{tabular}}{{@{{}}{alignments}@{{}}}}",
            r"            \toprule",
        ]
        body.extend("            " + line for line in latex_rows)
        body.extend([
            r"            \bottomrule",
            r"        \end{tabular}",
            r"    }",
            r"\end{table}",
        ])
        return "\n".join(body)

    body = [
        r"% Requires: \usepackage{booktabs,multirow,graphicx}",
        rf"\begin{{{environment}}}[t]",
        r"    \centering",
        r"    \caption{",
        f"        {caption}",
        r"    }",
        rf"    \label{{{label}}}",
    ]
    if wide and resize_wide:
        body.extend([
            r"    \resizebox{.98\textwidth}{!}{%",
            rf"        \begin{{tabular}}{{@{{}}{alignments}@{{}}}}",
            r"            \toprule",
        ])
        body.extend("            " + line for line in latex_rows)
        body.extend([
            r"            \bottomrule",
            r"        \end{tabular}",
            r"    }",
        ])
    else:
        body.extend([
            rf"    \begin{{tabular}}{{@{{}}{alignments}@{{}}}}",
            r"        \toprule",
        ])
        body.extend("        " + line for line in latex_rows)
        body.extend([
            r"        \bottomrule",
            r"    \end{tabular}",
        ])
    body.append(rf"\end{{{environment}}}")
    return "\n".join(body)
