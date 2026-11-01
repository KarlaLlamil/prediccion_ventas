"""Métricas del backtest (DECISIONES.md §13).

Cada fila de entrada es (SKU, origen) con el real `y`, el pronóstico de cada modelo y las escalas del SKU calculadas con
datos ≤ origen (error in-sample del seasonal-naive s = 12):
- MASE  = media por SKU de |e| / escala_mae; luego media entre SKUs.
- RMSSE = media entre SKUs de sqrt(media de e² / escala_mse).
- WAPE  = Σ|e| / Σy (agregado; legible para negocio).
- relMAE = Σ|e| / Σ|e del seasonal-naive| (< 1 → mejor que el seasonal-naive).
- Sesgo = Σe / Σy (> 0 sobre-pronóstico, < 0 sub-pronóstico).
- MAPE solo sobre y > 0, como referencia (LOG #8).
- Total: se suman SKUs por mes objetivo y se mide WAPE y sesgo del total (vista de compras).

Métrica de decisión: **WAPE + sesgo** (MASE/RMSSE se reportan como secundarias). Regla de adopción (`adopta`): un
candidato reemplaza a la referencia solo si baja el WAPE en ≥ 0.5 pp, gana en WAPE en ≥ 2/3 de los orígenes y no
empeora el |sesgo| en más de 1 pp.
"""
import numpy as np
import pandas as pd


def escalas(y, i_nac, k, o):
    """Escalas MASE/RMSSE por SKU (filas k) con datos desde el nacimiento hasta el origen o."""
    d = y[k, 12:o + 1] - y[k, :o - 11]
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(d), axis=1), np.nanmean(d ** 2, axis=1)


def clase_sb(adi, cv2):
    """Clasificación Syntetos–Boylan (umbrales de §11) calculada en el origen."""
    cv2 = np.nan_to_num(cv2, nan=0.0)
    return np.select([(adi < 1.32) & (cv2 < .49), adi < 1.32, cv2 < .49], ["suave", "errática", "intermitente"], "grumosa")


def resumen(df, modelos, ref="SNaive"):
    filas = {}
    e_ref = (df[ref] - df.y).abs().sum()
    for m in modelos:
        e = df[m] - df.y
        ok = df.escala_mae > 0
        mase = (e.abs() / df.escala_mae)[ok].groupby(df.id_articulo[ok]).mean().mean()
        ok2 = df.escala_mse > 0
        rmsse = np.sqrt((e ** 2 / df.escala_mse)[ok2].groupby(df.id_articulo[ok2]).mean()).mean()
        pos = df.y > 0
        tot = df.groupby("mes_obj")[[m, "y"]].sum()
        filas[m] = {"MASE": mase, "RMSSE": rmsse, "WAPE": e.abs().sum() / df.y.sum(),
                    "relMAE": e.abs().sum() / e_ref, "sesgo": e.sum() / df.y.sum(),
                    "MAPE_y>0": (e.abs() / df.y)[pos].mean(),
                    "WAPE_total": (tot[m] - tot.y).abs().sum() / tot.y.sum(),
                    "sesgo_total": (tot[m] - tot.y).sum() / tot.y.sum()}
    return pd.DataFrame(filas).T


MEJORA_WAPE, FRAC_ORIGENES, TOL_SESGO = .005, 2 / 3, .01


def wape(df, modelos):
    return pd.Series({m: (df[m] - df.y).abs().sum() / df.y.sum() for m in modelos})


def sesgo(df, modelos):
    return pd.Series({m: (df[m] - df.y).sum() / df.y.sum() for m in modelos})


def por_origen(df, modelos, metrica=wape):
    """Métrica por origen (filas) y modelo (columnas)."""
    return pd.DataFrame({o: metrica(g, modelos) for o, g in df.groupby("origen")}).T


def adopta(df, cand, ref):
    """Regla de adopción WAPE + sesgo (docstring del módulo)."""
    w, s = wape(df, [cand, ref]), sesgo(df, [cand, ref]).abs()
    po = por_origen(df, [cand, ref])
    return bool(w[cand] <= w[ref] - MEJORA_WAPE and (po[cand] < po[ref]).mean() >= FRAC_ORIGENES
                and s[cand] <= s[ref] + TOL_SESGO)


def elegir(df, candidatos, ref):
    """De los candidatos que cumplen `adopta` frente a `ref`, el de menor WAPE; si ninguno, `ref`."""
    ok = [c for c in candidatos if c != ref and adopta(df, c, ref)]
    return min(ok, key=lambda c: wape(df, [c])[c]) if ok else ref
