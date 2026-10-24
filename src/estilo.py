"""Estilo común para las gráficas de los notebooks (matplotlib).

Uso:
    from estilo import aplicar, COLORES, AZUL
    aplicar()
"""
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

# Paleta categórica validada (orden fijo: nunca se cicla; más de 8 series → agrupar en "Otros")
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
AZUL = COLORES[0]
GRIS = "#898781"          # ejes, etiquetas secundarias, datos de contexto
GRIS_CLARO = "#c3c2b7"    # dato parcial / atenuado
# Rampa secuencial azul (claro → oscuro) para mapas de calor
AZULES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def aplicar():
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": "#0b0b0b",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=COLORES),
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "xtick.color": GRIS,
        "ytick.color": GRIS,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    })


def miles(x, _=None):
    """Formato de eje: 1,234,567 → '1.2 M' / '850 k'."""
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f} M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f} k"
    return f"{x:.0f}"


FMT_MILES = FuncFormatter(miles)
