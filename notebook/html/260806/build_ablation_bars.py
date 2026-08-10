#!/usr/bin/env python3
"""Build the matched bar charts for Sections 1.1–1.3."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
MEETING = HERE / "260806_meeting.html"

# Metric order: Pearson r, Spearman ρ, RMSE.
MLP_METRICS = (
    ("Test Pearson r", "test_r", 0.63, 0.67, (0.63, 0.65, 0.67), False),
    ("Test Spearman ρ", "test_rho", 0.61, 0.65, (0.61, 0.63, 0.65), False),
    ("Test RMSE", "rmse", 1.34, 1.54, (1.34, 1.44, 1.54), True),
)

LOSS_METRICS = (
    ("Test Pearson r", "test_r", 0.654, 0.668, (0.655, 0.661, 0.667), False),
    ("Test Spearman ρ", "test_rho", 0.638, 0.650, (0.638, 0.644, 0.650), False),
    ("Test RMSE", "rmse", 1.32, 1.38, (1.32, 1.35, 1.38), True),
)

MFODFC_METRICS = (
    ("Test Pearson r", "test_r", 0.58, 0.67, (0.58, 0.625, 0.67), False),
    ("Test Spearman ρ", "test_rho", 0.57, 0.66, (0.57, 0.615, 0.66), False),
    ("Test RMSE", "rmse", 1.32, 1.50, (1.32, 1.41, 1.50), True),
)

CHARTS = (
    (
        "MLP_BAR",
        "mlp",
        "MLP-head trial comparison",
        "The canonical C+D+G champion is best on test Spearman correlation.",
        MLP_METRICS,
        (
            ("Base", "#2f6f4f", {"test_r": 0.660, "test_rho": 0.644, "rmse": 1.349}),
            ("T1", "#9aafca", {"test_r": 0.650, "test_rho": 0.634, "rmse": 1.391}),
            ("T2", "#315f9b", {"test_r": 0.645, "test_rho": 0.625, "rmse": 1.508}),
        ),
        None,
    ),
    (
        "LOSS_BAR",
        "loss",
        "Loss-term trial comparison",
        "MSE + Pearson-correlation term (T1) is the best probe-head loss — highest r/ρ and lowest RMSE.",
        LOSS_METRICS,
        (
            ("Base", "#2f6f4f", {"test_r": 0.660, "test_rho": 0.644, "rmse": 1.349}),
            ("T1", "#315f9b", {"test_r": 0.665, "test_rho": 0.648, "rmse": 1.342}),
            ("T2", "#9aafca", {"test_r": 0.663, "test_rho": 0.647, "rmse": 1.360}),
            ("T3", "#9aafca", {"test_r": 0.657, "test_rho": 0.641, "rmse": 1.368}),
            ("T4", "#9aafca", {"test_r": 0.662, "test_rho": 0.644, "rmse": 1.375}),
        ),
        "T1",
    ),
    (
        "MFODFC_BAR",
        "mfodfc",
        "mFo–DFc channel trial comparison",
        "The C+D+G champion is best on every metric; Trial 2 is the best added-channel trial.",
        MFODFC_METRICS,
        (
            ("Base", "#2f6f4f", {"test_r": 0.660, "test_rho": 0.644, "rmse": 1.349}),
            ("T1", "#9aafca", {"test_r": 0.628, "test_rho": 0.611, "rmse": 1.431}),
            ("T2", "#315f9b", {"test_r": 0.633, "test_rho": 0.618, "rmse": 1.431}),
            ("T3", "#9aafca", {"test_r": 0.600, "test_rho": 0.588, "rmse": 1.477}),
        ),
        "T2",
    ),
)


def y_position(value: float, low: float, high: float, invert: bool) -> float:
    # Normal ascending axis for every panel (bar height = raw value, low at the
    # baseline). `invert` no longer flips the RMSE axis — the "lower is better"
    # cue lives in the panel subtitle instead.
    top, bottom = 68.0, 235.0
    ratio = (value - low) / (high - low)
    return bottom - ratio * (bottom - top)


def svg(chart_id: str, title: str, description: str, metrics: tuple, series: tuple,
        winner: str = None) -> str:
    # Three compact 160 px metric panels with minimal inter-panel whitespace.
    width, height = 502, 294
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-labelledby="{chart_id}-title {chart_id}-desc">',
        f'<title id="{chart_id}-title">{title}</title>',
        f'<desc id="{chart_id}-desc">{description}</desc>',
    ]

    # Compact layout: keep the chart full-width and minimize unused horizontal space.
    outer_margin = 3
    panel_gap = 8
    panel_width = (
        width - 2 * outer_margin - panel_gap * (len(metrics) - 1)
    ) / len(metrics)
    for panel_index, (metric_title, key, low, high, ticks, invert) in enumerate(metrics):
        panel_x = outer_margin + panel_index * (panel_width + panel_gap)
        axis_x = panel_x + 34
        plot_right = panel_x + panel_width - 7
        parts.extend(
            [
                f'<rect x="{panel_x:.1f}" y="7" width="{panel_width:.1f}" height="275" rx="10" '
                'fill="#fbfcfe" stroke="#dce3eb"/>',
                f'<text x="{panel_x + panel_width / 2:.1f}" y="31" text-anchor="middle" '
                'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                f'font-size="14" font-weight="700" fill="#17202b">{metric_title}</text>',
                f'<text x="{panel_x + panel_width / 2:.1f}" y="50" text-anchor="middle" '
                'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                'font-size="10" font-weight="650" fill="#657284">'
                f'{"lower is better" if invert else "higher is better"}</text>',
            ]
        )

        for tick in ticks:
            tick_y = y_position(tick, low, high, invert)
            parts.extend(
                [
                    f'<line x1="{axis_x}" y1="{tick_y:.1f}" x2="{plot_right}" y2="{tick_y:.1f}" '
                    'stroke="#e2e7ed" stroke-width="1"/>',
                    f'<text x="{axis_x - 5}" y="{tick_y + 3.5:.1f}" text-anchor="end" '
                    'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                    f'font-size="9" fill="#778396">{tick:.2f}</text>',
                ]
            )

        parts.append(
            f'<line x1="{axis_x}" y1="235" x2="{plot_right}" y2="235" '
            'stroke="#8995a4" stroke-width="1"/>'
        )

        plot_width = plot_right - axis_x - 4
        # Match Figure 1: narrow bars with a small, fixed gap, centered as one group.
        bar_width = 14.0
        bar_gap = 8.0
        cluster_width = bar_width * len(series) + bar_gap * (len(series) - 1)
        cluster_x = axis_x + 4 + (plot_width - cluster_width) / 2
        for series_index, (label, color, values) in enumerate(series):
            value = values[key]
            bar_x = cluster_x + series_index * (bar_width + bar_gap)
            center_x = bar_x + bar_width / 2
            stroke = "#183d68" if label == winner else ("#19472e" if label == "Base" else "#7188a7")
            if value is None:
                parts.extend(
                    [
                        f'<rect x="{bar_x:.1f}" y="207" width="{bar_width:.1f}" height="28" rx="3" '
                        'fill="none" stroke="#9aa3b2" stroke-width="1.2" stroke-dasharray="4 3"/>',
                        f'<text x="{center_x:.1f}" y="201" text-anchor="middle" '
                        'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                        'font-size="8.5" font-weight="650" fill="#7d8795">pending</text>',
                    ]
                )
            else:
                bar_y = y_position(value, low, high, invert)
                parts.extend(
                    [
                        f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                        f'height="{235 - bar_y:.1f}" rx="3" fill="{color}" stroke="{stroke}" '
                        'stroke-width="1.2"/>',
                        f'<text x="{center_x:.1f}" y="{max(64, bar_y - 6):.1f}" text-anchor="middle" '
                        'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                        f'font-size="8" font-weight="700" fill="#273241">{value:.3f}</text>',
                    ]
                )
            parts.append(
                f'<text x="{center_x:.1f}" y="253" text-anchor="middle" '
                'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                f'font-size="8.5" font-weight="650" fill="{stroke}">{label}</text>'
            )

    parts.append('</svg>')
    return "\n".join(parts)


def inject(html: str, marker: str, chart: str) -> str:
    start_marker = f"      <!-- {marker}_START -->"
    end_marker = f"      <!-- {marker}_END -->"
    start = html.index(start_marker) + len(start_marker)
    end = html.index(end_marker, start)
    embedded = "\n" + "\n".join(f"      {line}" for line in chart.splitlines()) + "\n"
    return html[:start] + embedded + html[end:]


def main() -> None:
    html = MEETING.read_text(encoding="utf-8")
    for marker, chart_id, title, description, metrics, series, winner in CHARTS:
        html = inject(
            html, marker, svg(chart_id, title, description, metrics, series, winner)
        )
    MEETING.write_text(html, encoding="utf-8")
    print(f"embedded three ablation charts in {MEETING}")


if __name__ == "__main__":
    main()
