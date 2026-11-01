"""Pronóstico jerárquico middle-out (LOG #32): total por familia = nivel × índice estacional, repartido entre SKUs.

En cada origen T (mismo walk-forward y set evaluado que `backtest.py`, h = 3):
    serie de familia  = Σ de la serie de modelado (`Datos.serie(T)`: limpia para suaves/erráticos, cruda para
                        grumosos/intermitentes) de los SKUs del set, hasta T
    índice estacional = `features.indice_estacional` de la familia con datos ≤ T (medianas, sin COVID);
                        si la familia no tiene índice completo, el del total del set
    nivel_k           = media de los últimos k meses de la serie desestacionalizada (serie / índice del mes)
    F_familia         = nivel_k × índice del mes objetivo
Reparto entre los SKUs de la familia:
    TD_hist   cuota histórica del SKU (Σ 12 meses del SKU / Σ 12 meses de la familia)
    TD_ML     cuota del pronóstico ML_base (ML_base del SKU / Σ ML_base de la familia): el ML decide el reparto,
              la familia decide el volumen
    TD_mix    como TD_ML, pero el volumen de la familia es el promedio de nivel × índice y Σ ML_base (combinación)
k se elige en validación con la regla WAPE + sesgo de TD_ML frente a k = 12 (`metricas.elegir`).

Regla `TD_estado` (LOG #45), por el estado de la familia en el origen (causal, venta cruda de toda la familia hasta
T): r = media de los 3 meses ≤ T / media de los 12 anteriores (T−14…T−3).
    caída   (r < 0.85)  → TD_mix_k3    el nivel de 3 meses corta el sobre-pronóstico tras una caída
    subida  (r > 1.15)  → TD_mix_k12   el nivel de 12 meses no persigue un alza que suele revertir
    estable             → ML_base
Modelo adoptado (LOG #50): **`TD_estado_LGBM_tw`** = la misma regla con LightGBM Tweedie en el lugar de `ML_base`
(rama estable, Σ y cuota de `TD_mix`). Se construye con `construir(datos, tabla.assign(ML_base=<LightGBM>))`.
Justificación y sensibilidad a los umbrales en `02_modelos` §3.6–3.8 (`src/regimen.py`).

Uso:  venv\\Scripts\\python.exe src\\jerarquico.py   (escribe data/processed/resultados/pronosticos_*_jer.parquet)
"""
import json

import numpy as np
import pandas as pd

import features as ft
from backtest import ORIGENES, SALIDA
from features import H, MESES, indice_estacional
from metricas import elegir, por_origen, resumen, wape

VENTANAS = [3, 6, 12]
UMBRAL_CAIDA, UMBRAL_SUBIDA = .85, 1.15
MODELO_ESTADO = {"caída": "TD_mix_k3", "subida": "TD_mix_k12"}     # estable → ML_base
ELEGIDO = "TD_estado"                  # columna de la regla, con el ML que traiga la tabla en `ML_base`
# Adoptado (LOG #50): la misma regla con LightGBM Tweedie como ML (`td_lgbm.py`, `06_td_lightgbm` §7, `07_inventario_lgbm`)
ADOPTADO = "TD_estado_LGBM_tw"
REGRESOR = "L_tweedie"                 # configuración de `comparar_lgbm.HIPER`


def modelo_elegido():
    """Pronóstico de los SKUs maduros que usan inventario y pedido (LOG #50; antes `TD_estado`, LOG #45)."""
    return ADOPTADO


def estado_familia(datos, bajo=UMBRAL_CAIDA, alto=UMBRAL_SUBIDA):
    """Estado (caída / estable / subida), racha y r por mes × familia (+ TOTAL), con venta cruda ≤ cada mes."""
    fams, fy = ft._por_familia(datos, datos.y)
    fy = pd.DataFrame(fy.T, index=MESES, columns=fams)
    fy["TOTAL"] = fy.sum(axis=1)
    r = fy.rolling(3).mean() / fy.rolling(12).mean().shift(3)
    est = pd.DataFrame(np.select([r < bajo, r > alto], ["caída", "subida"], "estable"), index=r.index, columns=r.columns)
    racha = est.apply(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
    return est, racha, r


def regla_estado(d, mapa=MODELO_ESTADO, ref="ML_base"):
    """Pronóstico según `d.estado` (estado de la familia en el origen); los estados fuera del mapa usan ref."""
    out = d[ref].to_numpy().copy()
    for e, m in mapa.items():
        out = np.where(d.estado == e, d[m], out)
    return out


def familia_origen(datos, d, o, ks=VENTANAS):
    """Pronóstico de cada familia del set en el origen o, para cada ventana k. Regresa DataFrame familia × k."""
    k_sku = datos.skus.get_indexer(d.id_articulo)
    serie = pd.DataFrame(np.nan_to_num(datos.serie(o)[k_sku, :o + 1]).T, index=MESES[:o + 1]) \
        .T.groupby(d.familia.to_numpy()).sum().T
    ie = indice_estacional(serie)
    ie_tot = indice_estacional(serie.sum(axis=1).to_frame("total"))["total"]
    ie = ie.apply(lambda c: c if c.notna().all() else ie_tot)
    mes_obj = (MESES[o] + pd.DateOffset(months=H)).month
    desest = serie / ie.loc[serie.index.month].to_numpy()
    return pd.DataFrame({f"k{k}": desest.iloc[-k:].mean() * ie.loc[mes_obj] for k in ks})


def construir(datos, d):
    """Agrega TD_* (para cada k) y TD_estado a un DataFrame de pronósticos del backtest."""
    est = estado_familia(datos)[0]
    out = []
    for origen, g in d.groupby("origen"):
        o = MESES.get_loc(origen)
        fam = familia_origen(datos, g, o)
        k_sku = datos.skus.get_indexer(g.id_articulo)
        h12 = np.nansum(datos.serie(o)[k_sku, o - 11:o + 1], axis=1)
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
        g["estado_fam"] = est.loc[origen].reindex(g.familia).to_numpy()
        g[ELEGIDO] = regla_estado(g.assign(estado=g.estado_fam))
        out.append(g)
    return pd.concat(out).sort_index()


def gana_por_origen(d, m, ref="ML_base"):
    """Fracción de orígenes donde m tiene menor WAPE que ref."""
    po = por_origen(d, [ref, m])
    return (po[m] < po[ref]).mean()


def main():
    datos = ft.Datos.cargar()
    res = {p: construir(datos, pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet")) for p in [*ORIGENES, "puente"]}
    val = res["validacion"]
    cands = [f"TD_ML_k{k}" for k in VENTANAS]
    k = elegir(val, cands, f"TD_ML_k{max(VENTANAS)}").split("_")[-1]
    modelos = ["ML_base", "SARIMA_fam", f"TD_hist_{k}", f"TD_ML_{k}", f"TD_mix_{k}", "TD_mix_k3", ELEGIDO]
    modelos = list(dict.fromkeys(modelos))
    meta = {"k_elegido": k, "WAPE_por_k": wape(val, cands).to_dict(), "modelo_elegido": ELEGIDO,
            "umbrales": {"caida": UMBRAL_CAIDA, "subida": UMBRAL_SUBIDA}, "modelo_por_estado": MODELO_ESTADO}
    for p, d in res.items():
        cols = ["id_articulo", "origen", "mes_obj", "estado_fam"] + \
               [c for c in d.columns if c.startswith(("TD_", "F_fam_"))]
        d[cols].to_parquet(SALIDA / f"pronosticos_{p}_jer.parquet", index=False)
        if p == "puente":
            continue
        print(f"\n{p}")
        print(resumen(d, modelos).round(3).to_string())
        meta[p] = {m: gana_por_origen(d, m) for m in modelos[1:]}
        print("orígenes donde mejora el WAPE de ML_base:", {m: round(v, 2) for m, v in meta[p].items()})
    (SALIDA / "meta_jer.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["WAPE_por_k"], indent=2))


if __name__ == "__main__":
    main()
