"""Typography shared by paper-figure exporters."""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager


def available_paper_font() -> str:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in (
        "Latin Modern Roman",
        "LM Roman 10",
        "Computer Modern Roman",
        "CMU Serif",
        "STIX Two Text",
        "DejaVu Serif",
    ):
        if candidate in available:
            return candidate
    return "serif"


PAPER_FONT = available_paper_font()


def configure_paper_style() -> str:
    """Configure Matplotlib to match the paper's LaTeX typography."""

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [PAPER_FONT],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    return PAPER_FONT
