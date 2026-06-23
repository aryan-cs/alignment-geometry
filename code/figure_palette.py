"""Canonical color tokens for generated figure images only.

These colors are intentionally scoped to generated image assets. They do not
control LaTeX, PDF hyperlink colors, or document theme styling.
"""

PURPLE = "#d073ff"
YELLOW = "#ffe373"
GREEN = "#9bff73"

PURPLE_D = "#8a2be2"
YELLOW_D = "#c79a0f"
GREEN_D = "#4caf2f"

INK = "#222222"
GRID = "#dddddd"
GREY = "#8a8a8a"
GREY_L = "#bbbbbb"
WHITE = "#ffffff"

PURPLE_RAMP = [WHITE, PURPLE, PURPLE_D]
GREEN_RAMP = [WHITE, GREEN, GREEN_D]

CANONICAL_FIGURE_HEXES = {
    PURPLE,
    YELLOW,
    GREEN,
    PURPLE_D,
    YELLOW_D,
    GREEN_D,
    INK,
    GRID,
    GREY,
    GREY_L,
    WHITE,
}
