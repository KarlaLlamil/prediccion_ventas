"""Pronóstico jerárquico middle-out (LOG #32): total por familia = nivel × índice estacional, repartido entre SKUs.

En cada origen T (mismo walk-forward y set evaluado que `backtest.py`, h = 4):
    serie de familia  = Σ de la serie limpia (tope causal de picos) de los SKUs del set, hasta T
    índice estacional = `features.indice_estacional` de la familia con datos ≤ T (medianas, sin COVID);
                        si la familia no tiene índice completo, el del total del set
    nivel_k           = media de los últimos k meses de la serie desestacionalizada (serie / índice del mes)
    F_familia         = nivel_k × índice del mes objetivo
Reparto entre los SKUs de la familia:
    TD_hist   cuota histórica del SKU (Σ 12 meses del SKU / Σ 12 meses de la familia)
    TD_ML     cuota del pronóstico ML_base (ML_base del SKU / Σ ML_base de la familia): el ML decide el reparto,
              la familia decide el volumen
    TD_mix    como TD_ML, pero el volumen de la familia es el promedio de nivel × índice y Σ ML_base (combinación)
k se elige en validación (MASE de TD_ML).

Uso:  venv\\Scripts\\python.exe src\\jerarquico.py   (escribe data/processed/resultados/pronosticos_*_jer.parquet)
"""
import json

import numpy as np
import pandas as pd

import features as ft
from backtest import ORIGENES, SALIDA
from features import H, MESES, indice_estacional
from metricas import resumen

VENTANAS = [3, 6, 12]


def familia_origen(datos, d, o, ks=VENTANAS):
    """Pronóstico de cada familia del set en el origen o, para cada ventana k. Regresa DataFrame familia × k."""
    k_sku = datos.skus.get_indexer(d.id_articulo)
    serie = pd.DataFrame(np.nan_to_num(datos.c[k_sku, :o + 1]).T, index=MESES[:o + 1]) \
        .T.groupby(d.familia.to_numpy()).sum().T
    ie = indice_estacional(serie)
    ie_tot = indice_estacional(serie.sum(axis=1).to_frame("total"))["total"]
    ie = ie.apply(lambda c: c if c.notna().all() else ie_tot)
    mes_obj = (MESES[o] + pd.DateOffset(months=H)).month
    desest = serie / ie.loc[serie.index.month].to_numpy()
    return pd.DataFrame({f"k{k}": desest.iloc[-k:].mean() * ie.loc[mes_obj] for k in ks})


def construir(datos, d):
    """Agrega TD_* (para cada k) a un DataFrame de pronósticos del backtest."""
    out = []
    for origen, g in d.groupby("origen"):
        o = MESES.get_loc(origen)
        fam = familia_origen(datos, g, o)
        k_sku = datos.skus.get_indexer(g.id_articulo)
        h12 = np.nansum(datos.c[k_sku, o - 11:o + 1], axis=1)
        g = g.copy()
        tot_ml = g.groupby("familia").ML_base.transform("sum")
        tot_h = pd.Series(h12, index=g.index).groupby(g.familia).transform("sum")
        cuota_ml = np.where(tot_ml > 0, g.ML_base / tot_ml, 0)
        cuota_h = np.where(tot_h > 0, h12 / tot_h, 0)
        for k in fam.columns:
            F = g.familia.map(fam[k]).fillna(tot_ml)
            F = np.where(F > 0, F, tot_ml)                    # familia sin nivel → el total del ML
            g[f"TD_hist_{k}"] = F * cuota_h
            g[f"TD_ML_{k}"] = F * cuota_ml
            g[f"TD_mix_{k}"] = (F + tot_ml) / 2 * cuota_ml
            g[f"F_fam_{k}"] = F
        out.append(g)
    return pd.concat(out).sort_index()


def gana_por_origen(d, m, ref="ML_base"):
    po = pd.DataFrame({o: resumen(g, [ref, m]).MASE for o, g in d.groupby("origen")}).T
    return (po[m] < po[ref]).mean()


def main():
    datos = ft.Datos.cargar()
    res = {p: construir(datos, pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet")) for p in ORIGENES}
    val = res["validacion"]
    k_ml = min((c for c in val.columns if c.startswith("TD_ML_")), key=lambda c: resumen(val, [c]).MASE.iloc[0])
    k = k_ml.split("_")[-1]
    modelos = ["ML_base", "SARIMA_fam", f"TD_hist_{k}", f"TD_ML_{k}", f"TD_mix_{k}"]
    meta = {"k_elegido": k, "MASE_por_k": {c: resumen(val, [c]).MASE.iloc[0]
                                          for c in val.columns if c.startswith("TD_ML_")}}
    for p, d in res.items():
        cols = ["id_articulo", "origen", "mes_obj"] + [c for c in d.columns if c.startswith(("TD_", "F_fam_"))]
        d[cols].to_parquet(SALIDA / f"pronosticos_{p}_jer.parquet", index=False)
        print(f"\n{p}")
        print(resumen(d, modelos).round(3).to_string())
        meta[p] = {m: gana_por_origen(d, m) for m in modelos[1:]}
        print("orígenes donde mejora a ML_base:", {m: round(v, 2) for m, v in meta[p].items()})
    (SALIDA / "meta_jer.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["MASE_por_k"], indent=2))


if __name__ == "__main__":
    main()
