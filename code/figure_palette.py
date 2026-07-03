"""Canonical color tokens for generated figure images only.

These colors are intentionally scoped to generated image assets. They do not
control LaTeX, PDF hyperlink colors, or document theme styling.
"""

TURBO_VIOLET = "#33184a"  # turbo(2 / 255)
TURBO_BLUE = "#4661d6"    # turbo(28 / 255)
TURBO_CYAN = "#2eb4f2"    # turbo(61 / 255)
TURBO_GREEN = "#3cf58e"   # turbo(99 / 255)
TURBO_YELLOW = "#f8be39"  # turbo(168 / 255)
TURBO_ORANGE = "#ec530f"  # turbo(207 / 255)
TURBO_RED = "#9b0f01"     # turbo(245 / 255)

# Compatibility aliases used by the figure generators. Each value is sampled
# directly from Matplotlib's turbo colormap.
PRIMARY_GREEN = TURBO_BLUE
PRIMARY_GREEN_D = TURBO_BLUE
CONTROL_RED = TURBO_RED
CONTROL_RED_D = TURBO_RED
ACCENT_BLUE = TURBO_ORANGE
ACCENT_BLUE_D = TURBO_ORANGE
ACCENT_ORANGE_D = TURBO_GREEN
ACCENT_PURPLE_D = TURBO_VIOLET
ACCENT_CYAN_D = TURBO_CYAN
ACCENT_YELLOW_D = TURBO_YELLOW

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
PRIMARY_GREEN_RAMP = TURBO_RAMP
ACCENT_BLUE_RAMP = TURBO_RAMP

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
    ACCENT_YELLOW_D,
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
