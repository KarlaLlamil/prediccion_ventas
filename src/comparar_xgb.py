"""Comparación del XGBoost por serie (`notebooks/XGBoost.py`, LOG #33) contra nuestros modelos, mismo walk-forward h = 4.

- SKU: los top 30 SKUs, solo los pares (SKU, origen) que están en el set evaluado del backtest.
- Familia: las top 30 familias. Real = total de la familia (todos sus SKUs). Lo nuestro se agrega por familia como lo
  usa la política de pedidos: maduros (`ML_base` o `TD_mix_k3`) + jóvenes (`Media_post`). Los SKUs pre-lanzamiento
  (~0.4 % de las unidades) no tienen pronóstico, igual que en producción. SNaive = el mismo mes del año anterior.
MASE con la escala de cada serie (MAE in-sample del seasonal-naive hasta el origen), como en `metricas.py`.
"""
import numpy as np
import pandas as pd

import features as ft
from backtest import SALIDA
from features import MESES
from metricas import escalas, resumen

NUESTROS = ["ML_base", "TD_mix_k3", "Ruta_clasica", "MA12", "SNaive"]


def _pronosticos(p):
    d = pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet")
    jer = pd.read_parquet(SALIDA / f"pronosticos_{p}_jer.parquet")[["id_articulo", "origen", "TD_mix_k3"]]
    return d.merge(jer, on=["id_articulo", "origen"])


def extras(d, m):
    """MAE, RMSE, sMAPE y R² fuera de muestra (las métricas del archivo original)."""
    e = d[m] - d.y
    den = ((d.y.abs() + d[m].abs()) / 2).replace(0, 1e-10)
    return {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "sMAPE": (e.abs() / den).mean(),
            "R2": 1 - (e ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()}


def tabla(d, modelos):
    r = resumen(d, modelos)[["MASE", "WAPE", "sesgo"]]
    return r.join(pd.DataFrame({m: extras(d, m) for m in modelos}).T)


def r2(y, f):
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - ((f - y) ** 2).sum() / sst if sst > 0 else np.nan


def r2_descompuesto(d, modelos):
    """R² agrupado (todas las series juntas) contra R² dentro de cada serie, y la referencia "oráculo": predecir
    para cada serie su propia media real del periodo (constante). Muestra cuánto del R² agrupado sale solo de que
    las series tienen tamaños distintos."""
    filas = {}
    oraculo = d.groupby("id_articulo").y.transform("mean")
    for m in modelos + ["oráculo"]:
        f = oraculo if m == "oráculo" else d[m]
        por_serie = pd.Series({s: r2(g.y, f[g.index]) for s, g in d.groupby("id_articulo")})
        filas[m] = {"R2_agrupado": r2(d.y, f), "R2_por_serie_mediana": por_serie.median(),
                    "%_series_R2<0": (por_serie < 0).mean()}
    return pd.DataFrame(filas).T


def nivel_sku():
    xg = pd.read_parquet(SALIDA / "xgb_top30_sku.parquet")
    out = {}
    for p in ("validacion", "prueba"):
        d = _pronosticos(p).merge(xg[xg.periodo == p][["id_articulo", "origen", "XGBoost"]], on=["id_articulo", "origen"])
        out[p] = d
    return out


def nivel_familia():
    datos = ft.Datos.cargar()
    fams, fy = ft._por_familia(datos, datos.y)
    pos = pd.Index(fams)
    xg = pd.read_parquet(SALIDA / "xgb_top30_familia.parquet")
    top = xg.familia.unique()
    out = {}
    for p in ("validacion", "prueba"):
        d = _pronosticos(p)
        jov = pd.read_parquet(SALIDA / f"cold_jovenes_{p}.parquet")[["id_articulo", "origen", "Media_post"]]
        jov["familia"] = pd.Series(datos.familia, index=datos.skus).reindex(jov.id_articulo).to_numpy()
        mad = d.groupby(["familia", "origen"])[["ML_base", "TD_mix_k3", "Ruta_clasica", "MA12"]].sum()
        joven = jov.groupby(["familia", "origen"]).Media_post.sum()
        f = xg[xg.periodo == p].set_index(["familia", "origen"])[["mes_obj", "XGBoost"]]
        f = f.join(mad).join(joven).fillna({"Media_post": 0})
        for m in ["ML_base", "TD_mix_k3", "Ruta_clasica", "MA12"]:
            f[m] = f[m].fillna(0) + f.Media_post
        f = f.reset_index()
        i = pos.get_indexer(f.familia)
        o = MESES.get_indexer(f.origen)
        obj = o + ft.H
        f["y"] = fy[i, obj]
        f["SNaive"] = fy[i, obj - 12]
        sc = [escalas(fy, None, [a], b) for a, b in zip(i, o)]
        f["escala_mae"] = [s[0][0] for s in sc]
        f["escala_mse"] = [s[1][0] for s in sc]
        out[p] = f.rename(columns={"familia": "id_articulo"}).assign(familia=lambda x: x.id_articulo)
    return out, top
