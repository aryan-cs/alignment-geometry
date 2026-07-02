"""Canonical color tokens for generated figure images only.

These colors are intentionally scoped to generated image assets. They do not
control LaTeX, PDF hyperlink colors, or document theme styling.
"""

PRIMARY_GREEN = "#11f568"
CONTROL_RED = "#f51152"
ACCENT_BLUE = "#118cf5"

PRIMARY_GREEN_D = "#08a84a"
CONTROL_RED_D = "#d80f49"
ACCENT_BLUE_D = "#087dd1"
ACCENT_ORANGE_D = "#d9650b"
ACCENT_PURPLE_D = "#8c42d1"
ACCENT_CYAN_D = "#0797a8"

INK = "#222222"
GRID = "#dddddd"
GREY = "#8a8a8a"
GREY_L = "#bbbbbb"
WHITE = "#ffffff"

PRIMARY_GREEN_RAMP = [WHITE, PRIMARY_GREEN, PRIMARY_GREEN_D]
ACCENT_BLUE_RAMP = [WHITE, ACCENT_BLUE, ACCENT_BLUE_D]

CANONICAL_FIGURE_HEXES = {
    PRIMARY_GREEN,
    CONTROL_RED,
    ACCENT_BLUE,
    PRIMARY_GREEN_D,
    CONTROL_RED_D,
    ACCENT_BLUE_D,
    ACCENT_ORANGE_D,
    ACCENT_PURPLE_D,
    ACCENT_CYAN_D,
    INK,
    GRID,
    GREY,
    GREY_L,
    WHITE,
}
