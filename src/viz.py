"""Shared matplotlib styling: validated categorical palette, thin marks, recessive grid."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Official CPCB AQI category colours (domain convention, always shown with labels)
AQI_COLORS = {"Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#ffd700",
              "Poor": "#ff9900", "Very Poor": "#ff0000", "Severe": "#a52a2a"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
