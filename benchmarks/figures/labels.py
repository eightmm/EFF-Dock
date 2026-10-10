"""Numerical mean labels for manuscript bars and interval plots."""

import math

DARK = "#333840"


def mean_label(ax, position, value, sd=0, *, digits=1, fontsize=8.5, dx=0):
    if value is None:
        return
    if not math.isfinite(value):
        raise ValueError("A numerical figure label requires a finite mean")
    ax.annotate(
        f"{value:.{digits}f}",
        (position, value + (sd or 0)),
        xytext=(dx, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=DARK,
        zorder=5,
    )


def success_labels(
    ax, position, rmsd, valid, rmsd_sd=0, valid_sd=0, *, horizontal=False, outside_dx=0
):
    # Both endpoint SDs can extend beyond the total-height mean.
    upper = max(rmsd + (rmsd_sd or 0), valid + (valid_sd or 0))
    if horizontal:
        ax.annotate(
            f"{rmsd:.1f}",
            (upper, position),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=DARK,
            zorder=5,
        )
        # Short bars centre the label inside the bar, clear of both error bars.
        short = valid < 20
        ax.text(
            valid / 2 if short else max(valid / 2, valid - (valid_sd or 0) - 3),
            position,
            f"{valid:.1f}",
            ha="center" if short else "right",
            va="center",
            fontsize=8.5,
            color=DARK,
            zorder=5,
        )
    else:
        mean_label(ax, position, rmsd, upper - rmsd, dx=outside_dx)
        ax.text(
            position,
            max(valid / 2, valid - (valid_sd or 0) - 9),
            f"{valid:.1f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=DARK,
            zorder=5,
        )


def interval_label(ax, row, value):
    ax.text(
        1.025,
        row,
        f"{value:+.2f}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8.5,
        color=DARK,
    )


def segment_labels(ax, positions, heights, bottoms, *, minimum=8):
    # Small segments remain visible without illegible or overlapping text.
    for position, height, bottom in zip(positions, heights, bottoms, strict=True):
        if height >= minimum:
            ax.text(
                position,
                bottom + height / 2,
                f"{height:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=DARK,
                zorder=5,
            )
