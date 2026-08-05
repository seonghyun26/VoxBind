#!/usr/bin/env python3
"""Build the matched bar charts for Sections 1.1–1.3."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
MEETING = HERE / "260806_meeting.html"

MLP_METRICS = (
    ("Best val ρ", "val_rho", 0.50, 0.58, (0.50, 0.54, 0.58), False),
    ("Test Pearson r", "test_r", 0.63, 0.67, (0.63, 0.65, 0.67), False),
    ("Test Spearman ρ", "test_rho", 0.61, 0.65, (0.61, 0.63, 0.65), False),
    ("Test RMSE", "rmse", 1.36, 1.54, (1.36, 1.45, 1.54), True),
)

MFODFC_METRICS = (
    ("Best val ρ", "val_rho", 0.48, 0.56, (0.48, 0.52, 0.56), False),
    ("Test Pearson r", "test_r", 0.58, 0.67, (0.58, 0.625, 0.67), False),
    ("Test Spearman ρ", "test_rho", 0.57, 0.66, (0.57, 0.615, 0.66), False),
    ("Test RMSE", "rmse", 1.32, 1.50, (1.32, 1.41, 1.50), True),
)

CHARTS = (
    (
        "MLP_BAR",
        "mlp",
        "MLP-head trial comparison",
        "The matched mean-pool champion is best on test Spearman correlation. Trial 3 is pending.",
        MLP_METRICS,
        (
            ("Base", "#2f6f4f", {"val_rho": 0.546, "test_r": 0.656, "test_rho": 0.641, "rmse": 1.420}),
            ("T1", "#9aafca", {"val_rho": 0.563, "test_r": 0.650, "test_rho": 0.634, "rmse": 1.391}),
            ("T2", "#315f9b", {"val_rho": 0.527, "test_r": 0.645, "test_rho": 0.625, "rmse": 1.508}),
            ("T3", "#9aafca", {"val_rho": None, "test_r": None, "test_rho": None, "rmse": None}),
        ),
    ),
    (
        "LOSS_BAR",
        "loss",
        "Loss-term trial comparison",
        "The matched MSE champion is shown; Trials 1 through 4 are pending.",
        MLP_METRICS,
        (
            ("Base", "#2f6f4f", {"val_rho": 0.546, "test_r": 0.656, "test_rho": 0.641, "rmse": 1.420}),
            ("T1", "#9aafca", {"val_rho": None, "test_r": None, "test_rho": None, "rmse": None}),
            ("T2", "#315f9b", {"val_rho": None, "test_r": None, "test_rho": None, "rmse": None}),
            ("T3", "#9aafca", {"val_rho": None, "test_r": None, "test_rho": None, "rmse": None}),
            ("T4", "#9aafca", {"val_rho": None, "test_r": None, "test_rho": None, "rmse": None}),
        ),
    ),
    (
        "MFODFC_BAR",
        "mfodfc",
        "mFo–DFc channel trial comparison",
        "The C+D+G champion is best on every metric; Trial 2 is the best added-channel trial.",
        MFODFC_METRICS,
        (
            ("Base", "#2f6f4f", {"val_rho": 0.546, "test_r": 0.660, "test_rho": 0.644, "rmse": 1.349}),
            ("T1", "#9aafca", {"val_rho": 0.523, "test_r": 0.628, "test_rho": 0.611, "rmse": 1.431}),
            ("T2", "#315f9b", {"val_rho": 0.535, "test_r": 0.633, "test_rho": 0.618, "rmse": 1.431}),
            ("T3", "#9aafca", {"val_rho": 0.502, "test_r": 0.600, "test_rho": 0.588, "rmse": 1.477}),
        ),
    ),
)


def y_position(value: float, low: float, high: float, invert: bool) -> float:
    top, bottom = 68.0, 235.0
    ratio = (value - low) / (high - low)
    return top + ratio * (bottom - top) if invert else bottom - ratio * (bottom - top)


def svg(chart_id: str, title: str, description: str, metrics: tuple, series: tuple) -> str:
    width, height = 1130, 294
    last_trial = series[-1][0]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-labelledby="{chart_id}-title {chart_id}-desc">',
        f'<title id="{chart_id}-title">{title}</title>',
        f'<desc id="{chart_id}-desc">{description}</desc>',
    ]

    for panel_index, (metric_title, key, low, high, ticks, invert) in enumerate(metrics):
        panel_x = 8 + panel_index * 280
        axis_x = panel_x + 39
        plot_right = panel_x + 254
        parts.extend(
            [
                f'<rect x="{panel_x}" y="7" width="266" height="275" rx="10" '
                'fill="#fbfcfe" stroke="#dce3eb"/>',
                f'<text x="{panel_x + 133}" y="31" text-anchor="middle" '
                'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                f'font-size="14" font-weight="700" fill="#17202b">{metric_title}</text>',
                f'<text x="{panel_x + 133}" y="50" text-anchor="middle" '
                'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                'font-size="10" font-weight="650" fill="#657284">'
                f'{"lower ↑" if invert else "higher ↑"}</text>',
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
        step = plot_width / len(series)
        bar_width = min(30.0, step * 0.58)
        for series_index, (label, color, values) in enumerate(series):
            value = values[key]
            center_x = axis_x + 4 + step * (series_index + 0.5)
            bar_x = center_x - bar_width / 2
            stroke = "#183d68" if label == "T2" else ("#19472e" if label == "Base" else "#7188a7")
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
                        f'font-size="9.5" font-weight="700" fill="#273241">{value:.3f}</text>',
                    ]
                )
            parts.append(
                f'<text x="{center_x:.1f}" y="253" text-anchor="middle" '
                'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
                f'font-size="9.5" font-weight="650" fill="{stroke}">{label}</text>'
            )

    parts.extend(
        [
            '<text x="565" y="278" text-anchor="middle" '
            'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
            f'font-size="10" fill="#657284">Base = champion · T1–{last_trial} = matched trials</text>',
            '</svg>',
        ]
    )
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
    for marker, chart_id, title, description, metrics, series in CHARTS:
        html = inject(html, marker, svg(chart_id, title, description, metrics, series))
    MEETING.write_text(html, encoding="utf-8")
    print(f"embedded three ablation charts in {MEETING}")


if __name__ == "__main__":
    main()
