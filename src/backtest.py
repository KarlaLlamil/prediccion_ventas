"""Backtest walk-forward con h = 4 (fase 3; DECISIONES.md §13, LOG_DECISIONES #19).

Periodos (meses objetivo; origen = objetivo − 4):
    validación  sep-2024 → ago-2025  (orígenes may-2024 → abr-2025): hiperparámetros, ablaciones, ruta por segmento.
    prueba      sep-2025 → sep-2026  (orígenes may-2025 → may-2026): solo se evalúa lo ya elegido.
En cada origen T todo se ajusta con datos ≤ T (ventana expansiva, reentrenamiento mensual).

Set evaluado en cada origen: SKUs vivos (venta en los últimos 12 meses) con ≥ 13 meses de historia. Los SKUs más
jóvenes (cold start) quedan fuera y se reporta su peso.

Modelos:
    Naive, SNaive (s = 12), MA12                         directos de la matriz
    ETS (AutoETS s = 12), Croston, SBA, TSB              statsforecast, por SKU
    SARIMA_fam                                           AutoARIMA s = 12 por familia × cuota del SKU (12 m)
    ML_*                                                 HistGradientBoosting (Poisson) global sobre features/escala

Uso:  venv\\Scripts\\python.exe src\\backtest.py          (escribe data/processed/resultados/)
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, CrostonClassic, CrostonSBA, TSB

import features as ft
from features import BASE, DEVOL, H, MESES, SIN_LIMPIEZA, SOBRE_AGR, SOBRE_SKU
from metricas import clase_sb, escalas, resumen

warnings.filterwarnings("ignore")
SALIDA = ft.PROC / "resultados"
I = MESES.get_loc
ORIGENES = {"validacion": range(I(pd.Timestamp("2024-05-01")), I(pd.Timestamp("2025-04-01")) + 1),
            "prueba": range(I(pd.Timestamp("2025-05-01")), I(pd.Timestamp("2026-05-01")) + 1)}
SHOCK_COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"))   # objetivos excluidos del entrenamiento ML
MIN_MESES_FAM = 36

HIPER = {"A": dict(learning_rate=.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=50),
         "B": dict(learning_rate=.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=100),
         "C": dict(learning_rate=.03, max_iter=600, max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1.0)}
VARIANTES = {"ML_base": BASE,
             "ML_sin_limpieza": [SIN_LIMPIEZA.get(f, f) for f in BASE],
             "ML_sobre_sku": BASE + SOBRE_SKU,
             "ML_sobre_agr": BASE + SOBRE_AGR,
             "ML_sobre_todo": BASE + SOBRE_SKU + SOBRE_AGR,
             "ML_devol": BASE + DEVOL}
CANDIDATOS_RUTA = ["SNaive", "MA12", "ETS", "SARIMA_fam", "Croston", "SBA", "TSB"]


def larga(y, i_nac, k, o):
    """Series de los SKUs k desde su nacimiento hasta el origen o, en formato largo para statsforecast."""
    filas = [pd.DataFrame({"unique_id": j, "ds": MESES[i_nac[j]:o + 1], "y": y[j, i_nac[j]:o + 1]}) for j in k]
    return pd.concat(filas, ignore_index=True)


def estadisticos(datos, k, o):
    obj = o + H
    sf = StatsForecast(models=[AutoETS(season_length=12, alias="ETS"), CrostonClassic(alias="Croston"),
                               CrostonSBA(alias="SBA"), TSB(alpha_d=.1, alpha_p=.1, alias="TSB")], freq="MS", n_jobs=1)
    fc = sf.forecast(df=larga(datos.y, datos.i_nac, k, o), h=H)
    fc = fc[fc.ds == MESES[o] + pd.DateOffset(months=H)].set_index("unique_id").reindex(k)
    y = datos.y
    out = pd.DataFrame({"Naive": y[k, o], "SNaive": y[k, obj - 12], "MA12": np.nanmean(y[k, o - 11:o + 1], axis=1)})
    for m in ["ETS", "Croston", "SBA", "TSB"]:
        out[m] = np.clip(fc[m].to_numpy(), 0, None)
    return out


def sarima_familia(datos, k, o):
    """AutoARIMA estacional por familia y desagregación por la cuota del SKU en los últimos 12 meses."""
    fams, fy = ft._por_familia(datos, datos.y)
    pos = pd.Index(fams).get_indexer(datos.familia[k])
    usadas = np.unique(pos)
    filas = []
    for p in usadas:
        s = fy[p, :o + 1]
        ini = np.argmax(s > 0)
        if o + 1 - ini >= MIN_MESES_FAM:
            filas.append(pd.DataFrame({"unique_id": p, "ds": MESES[ini:o + 1], "y": s[ini:]}))
    pron = pd.Series(np.nan, index=usadas)
    if filas:
        fc = StatsForecast(models=[AutoARIMA(season_length=12, alias="SARIMA")], freq="MS", n_jobs=1).forecast(
            df=pd.concat(filas), h=H)
        fc = fc[fc.ds == MESES[o] + pd.DateOffset(months=H)].set_index("unique_id").SARIMA
        pron.loc[fc.index] = np.clip(fc.to_numpy(), 0, None)
    falta = pron.isna()
    pron[falta] = fy[pron.index[falta], o + H - 12]                 # familia corta → seasonal-naive de la familia
    cuota = np.nansum(datos.y[k, o - 11:o + 1], axis=1) / np.maximum(fy[pos, o - 11:o + 1].sum(axis=1), 1e-9)
    return pron.reindex(pos).to_numpy() * cuota


def _matriz(tab, cols, categorias):
    x = tab[cols].copy()
    if "familia" in cols:
        x["familia"] = pd.Categorical(x.familia, categories=categorias)
    return x


def ml(tabla, cols, o, hiper):
    """Ajusta con objetivos conocidos en o (mes_obj ≤ T, sin el choque COVID) y pronostica las filas del origen o."""
    T = MESES[o]
    tr = tabla[(tabla.mes_obj <= T) & ~tabla.mes_obj.between(*SHOCK_COVID)]
    te = tabla[tabla.origen == T]
    cats = sorted(tr.familia.unique())
    m = HistGradientBoostingRegressor(loss="poisson", categorical_features="from_dtype", random_state=0, **hiper)
    m.fit(_matriz(tr, cols, cats), tr.y / tr.escala)
    return pd.Series(m.predict(_matriz(te, cols, cats)) * te.escala.to_numpy(), index=te.id_articulo.to_numpy())


def base_origen(datos, tabla, o):
    """Filas evaluables del origen o con real, escalas, clase SB y modelos estadísticos."""
    te = tabla[tabla.origen == MESES[o]]
    k = datos.skus.get_indexer(te.id_articulo)
    mae, mse = escalas(datos.y, datos.i_nac, k, o)
    df = pd.DataFrame({"id_articulo": te.id_articulo.to_numpy(), "familia": te.familia.to_numpy(),
                       "perfil": te.perfil.to_numpy(), "origen": MESES[o], "mes_obj": te.mes_obj.to_numpy(),
                       "y": te.y.to_numpy(), "escala_mae": mae, "escala_mse": mse,
                       "clase": clase_sb(te.adi.to_numpy(), te.cv2.to_numpy())})
    df = pd.concat([df, estadisticos(datos, k, o)], axis=1)
    df["SARIMA_fam"] = sarima_familia(datos, k, o)
    return df


def cobertura(datos, o):
    """Peso en el mes objetivo de los SKUs que no entran al set (jóvenes o nacidos después del origen)."""
    obj = o + H
    vivo = (datos.i_nac <= o - (ft.MIN_HISTORIA - 1)) & (np.nansum(datos.y[:, o - 11:o + 1], axis=1) > 0)
    tot = np.nansum(datos.y[:, obj])
    return {"mes_obj": MESES[obj], "u_total": tot, "u_fuera_set": np.nansum(datos.y[~vivo, obj])}


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    datos = ft.Datos.cargar()
    tabla = ft.construir(datos, list(range(12, len(MESES))))
    tabla.to_parquet(ft.PROC / "features.parquet", index=False)
    print(f"features: {tabla.shape}  ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    for periodo, origenes in ORIGENES.items():
        partes = []
        for o in origenes:
            t = time.time()
            partes.append(base_origen(datos, tabla, o))
            print(f"  {periodo} {MESES[o]:%Y-%m}: estadísticos ({time.time() - t:.0f}s)", flush=True)
        res[periodo] = pd.concat(partes, ignore_index=True)

    # 1) hiperparámetros del modelo global: solo validación, features base
    val = res["validacion"]
    hiper_res = {}
    for nombre, h in HIPER.items():
        p = pd.concat([ml(tabla, BASE, o, h) for o in ORIGENES["validacion"]])
        val[f"hp_{nombre}"] = p.to_numpy()
        hiper_res[nombre] = resumen(val, [f"hp_{nombre}"]).iloc[0].to_dict()
        print(f"  hiper {nombre}: MASE {hiper_res[nombre]['MASE']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    mejor = min(hiper_res, key=lambda n: hiper_res[n]["MASE"])
    val.drop(columns=[c for c in val.columns if c.startswith("hp_")], inplace=True)

    # 2) variantes de features (ablaciones), con los hiperparámetros elegidos, en validación y prueba
    for periodo, origenes in ORIGENES.items():
        for nombre, cols in VARIANTES.items():
            res[periodo][nombre] = pd.concat([ml(tabla, cols, o, HIPER[mejor]) for o in origenes]).to_numpy()
            print(f"  {periodo} {nombre} ({time.time() - t0:.0f}s)", flush=True)

    # 3) ruta por segmento: el mejor modelo clásico de cada clase SB en validación (por MASE) se aplica en prueba
    ruta = {c: resumen(g, CANDIDATOS_RUTA).MASE.idxmin() for c, g in val.groupby("clase")}
    for periodo in res:
        d = res[periodo]
        d["Ruta_clasica"] = [d.at[i, ruta.get(c, "SNaive")] for i, c in zip(d.index, d.clase)]

    for periodo, d in res.items():
        d.to_parquet(SALIDA / f"pronosticos_{periodo}.parquet", index=False)
    cob = pd.DataFrame([cobertura(datos, o) for p in ORIGENES.values() for o in p])
    cob.to_csv(SALIDA / "cobertura.csv", index=False)
    meta = {"hiper": hiper_res, "hiper_elegido": mejor, "ruta": ruta, "segundos": round(time.time() - t0)}
    (SALIDA / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
