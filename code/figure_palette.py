"""Canonical color tokens for generated figure images only.

These colors are intentionally scoped to generated image assets. They do not
control LaTeX, PDF hyperlink colors, or document theme styling.
"""

TURBO_VIOLET = "#38276d"  # turbo(0.03)
TURBO_BLUE = "#477bf2"    # turbo(0.15)
TURBO_CYAN = "#1ecbda"    # turbo(0.28)
TURBO_GREEN = "#46f884"   # turbo(0.40)
TURBO_YELLOW = "#e1dd37"  # turbo(0.60)
TURBO_ORANGE = "#fb7e21"  # turbo(0.75)
TURBO_RED = "#dd3d08"     # turbo(0.85)

SAFE_GREEN = TURBO_GREEN
HARM_RED = TURBO_RED

INK = "#222222"
GRID = "#dddddd"
GREY = "#8a8a8a"
GREY_L = "#bbbbbb"
WHITE = "#ffffff"

TURBO_RAMP = [
    TURBO_VIOLET,
    TURBO_BLUE,
    TURBO_CYAN,
    TURBO_GREEN,
    TURBO_YELLOW,
    TURBO_ORANGE,
    TURBO_RED,
]
CANONICAL_FIGURE_HEXES = {
    TURBO_VIOLET,
    TURBO_BLUE,
    TURBO_CYAN,
    TURBO_GREEN,
    TURBO_YELLOW,
    TURBO_ORANGE,
    TURBO_RED,
    INK,
    GRID,
    GREY,
    GREY_L,
    WHITE,
}
