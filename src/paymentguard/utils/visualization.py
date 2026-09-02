import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "bg": "#0f1117",
    "card": "#1a1d27",
    "accent_red": "#ff4757",
    "accent_orange": "#ffa502",
    "accent_green": "#2ed573",
    "accent_blue": "#1e90ff",
    "accent_purple": "#a29bfe",
    "text": "#e0e0e0",
    "subtext": "#8888aa",
    "grid": "#2a2d3e",
}


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["card"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["subtext"],
        "ytick.color": PALETTE["subtext"],
        "text.color": PALETTE["text"],
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "legend.facecolor": PALETTE["card"],
        "legend.edgecolor": PALETTE["grid"],
        "figure.autolayout": True,
    })
