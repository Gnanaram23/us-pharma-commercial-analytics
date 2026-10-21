"""One chart style for every project, so the figures read as a set."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
ACCENT = "#0F766E"      # teal: the series the chart is about
ACCENT_2 = "#B45309"    # amber: the comparison or the warning
NEUTRAL = "#9CA3AF"
PALETTE = ["#0F766E", "#2563EB", "#B45309", "#7C3AED", "#DC2626", "#0891B2", "#65A30D", "#DB2777", "#4B5563"]


def apply() -> None:
    mpl.rcParams.update({
        "text.parse_math": False,   # so '$' in labels prints as a dollar sign
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def subtitle(ax, text: str) -> None:
    """Grey line under the title saying what the reader should notice.

    Call after ax.set_title(); the title is lifted to make room."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=26)
    ax.text(0, 1.015, text, transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom")


def source(fig, text: str) -> None:
    fig.text(0.01, -0.02, f"Source: {text}", color=MUTED, fontsize=8.5, ha="left", va="top")


def dollars(ax, axis="y", unit="B") -> None:
    div = {"B": 1e9, "M": 1e6, "K": 1e3}[unit]
    fmt = FuncFormatter(lambda v, _: f"${v / div:,.0f}{unit}" if float(v / div).is_integer() else f"${v / div:,.1f}{unit}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def pct(ax, axis="y", decimals=0) -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:.{decimals}%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, path) -> None:
    """Save the chart for the README and, inside a notebook, show it inline too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    try:
        get_ipython  # noqa: F821  (only defined inside Jupyter)
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except NameError:
        pass
    plt.close(fig)
