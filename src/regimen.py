"""Análisis de régimen y verificación de la regla de limpieza (LOG #42–#44). Resultados en `02_modelos` §3.4–3.7.

1. Ablación de la regla nueva (LOG #39): mismo `ML_base` y mismos orígenes, revirtiendo cada parte (objetivo
   crudo, limpieza para todas las clases, sin piso de meses bajos + COVID fijo, ADI hasta el origen, todo lo viejo).
   La evaluación siempre es contra la venta cruda.
2. ETS como estadístico de familia: igual que `jerarquico` pero con F_familia = AutoETS (s = 12) de la serie de la
   familia (≥ 36 meses; si no, el nivel de 12 meses). Variantes `TD_ML_ets`, `TD_mix_ets`, `TD_hist_ets`.
3. Estado de la familia **en el origen** (`jerarquico.estado_familia`, causal): r = media de los 3 meses ≤ T /
   media de los 12 anteriores (T−14…T−3), con la venta cruda de todos los SKUs de la familia. caída r < 0.85,
   subida r > 1.15, estable en medio; `racha` = meses seguidos en ese estado hasta T.
4. Regla por estado: A = `TD_mix_k3` si la familia está en caída, `ML_base` si no; B = además `TD_mix_k12` en subida
   (B es `TD_estado`, el modelo elegido, LOG #45). Sensibilidad a los umbrales en `sensibilidad`.

Uso:  venv\\Scripts\\python.exe src\\regimen.py   (escribe data/processed/resultados/regimen_*.parquet; ~7 min)
"""
import copy
import json
import time

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

import backtest as bt
import features as ft
import jerarquico as jq
from features import H, MESES

SALIDA = bt.SALIDA
PERIODOS = ["validacion", "puente", "prueba"]


# 1. ablación de la regla de limpieza
def _sin_piso(d):
    """Serie limpia sin el piso de meses bajos y exclusión fija de mar–jun 2020 (regla anterior)."""
    d = copy.copy(d)
    d.c, d.outlier = ft.limpiar_outliers(d.y)
    d.bajo = np.isin(MESES, pd.date_range("2020-03-01", "2020-06-01", freq="MS"))
    return d


def _todos_limpios(d):
    d = copy.copy(d)
    d.clase = np.where(d.clase == "", "", "suave").astype(object)
    return d


def _adi_viejo(t, d):
    """ADI con los meses desde el nacimiento hasta el origen (incluye los ceros posteriores a la última venta)."""
    n_nz = ft.clase_sb_causal(d.y, d.i_nac)[3]
    k, o = d.skus.get_indexer(t.id_articulo), MESES.get_indexer(t.origen)
    return t.assign(adi=(o - d.i_nac[k] + 1) / np.maximum(n_nz[k, o], 1))


def ablacion_regla(datos):
    hp = bt.HIPER[json.loads((SALIDA / "meta.json").read_text(encoding="utf-8"))["hiper_elegido"]]
    todos = list(range(12, len(MESES)))
    t_act = ft.construir(datos, todos)
    t_viejo = ft.construir(_todos_limpios(_sin_piso(datos)), todos)
    variantes = {
        "actual": t_act,
        "objetivo_crudo": t_act.assign(y_entreno=t_act.y),
        "limpieza_todos": ft.construir(_todos_limpios(datos), todos),
        "sin_piso_covid_fijo": ft.construir(_sin_piso(datos), todos),
        "adi_viejo": _adi_viejo(t_act, datos),
        "regla_vieja": _adi_viejo(t_viejo.assign(y_entreno=t_viejo.y), datos),
    }
    for p in ["validacion", "prueba"]:
        base = pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet")
        out = []
        for o in bt.ORIGENES[p]:
            te = base[base.origen == MESES[o]][["id_articulo", "familia", "origen", "mes_obj", "y"]].copy()
            for n, t in variantes.items():
                te[n] = bt.ml(t, bt.VARIANTES["ML_base"], o, hp).reindex(te.id_articulo).to_numpy()
            out.append(te)
        pd.concat(out, ignore_index=True).to_parquet(SALIDA / f"regimen_ablacion_{p}.parquet", index=False)


# 2. ETS como estadístico de familia
def ets_familia(serie, o):
    """AutoETS estacional por familia (serie: mes × familia, datos ≤ o); pronóstico de o + H, sin negativos."""
    filas = []
    for fam, s in serie.items():
        ini = np.argmax(s.to_numpy() > 0)
        if len(s) - ini >= bt.MIN_MESES_FAM:
            filas.append(pd.DataFrame({"unique_id": fam, "ds": s.index[ini:], "y": s.to_numpy()[ini:]}))
    if not filas:
        return pd.Series(dtype=float)
    fc = StatsForecast(models=[AutoETS(season_length=12, alias="ETS")], freq="MS", n_jobs=1).forecast(
        df=pd.concat(filas), h=H)
    return fc[fc.ds == MESES[o] + pd.DateOffset(months=H)].set_index("unique_id").ETS.clip(lower=0)


def td_ets(datos, d):
    """TD_* con F_familia = ETS, mismo reparto que `jerarquico.construir`."""
    out = []
    for origen, g in d.groupby("origen"):
        o = MESES.get_loc(origen)
        k_sku = datos.skus.get_indexer(g.id_articulo)
        serie = pd.DataFrame(np.nan_to_num(datos.serie(o)[k_sku, :o + 1]).T, index=MESES[:o + 1]) \
            .T.groupby(g.familia.to_numpy()).sum().T
        k12 = jq.familia_origen(datos, g, o, ks=[12]).k12
        F_fam = ets_familia(serie, o).reindex(k12.index).fillna(k12)
        h12 = np.nansum(datos.serie(o)[k_sku, o - 11:o + 1], axis=1)
        tot_ml = g.groupby("familia").ML_base.transform("sum")
        tot_h = pd.Series(h12, index=g.index).groupby(g.familia).transform("sum")
        cuota_ml = np.where(tot_ml > 0, g.ML_base / tot_ml, 0)
        cuota_h = np.where(tot_h > 0, h12 / tot_h, 0)
        F = g.familia.map(F_fam).fillna(tot_ml)
        F = np.where(F > 0, F, tot_ml)
        out.append(g[["id_articulo", "origen"]].assign(TD_hist_ets=F * cuota_h, TD_ML_ets=F * cuota_ml,
                                                        TD_mix_ets=(F + tot_ml) / 2 * cuota_ml, F_fam_ets=F))
    return pd.concat(out)


# 3–4. estado de la familia en el origen y regla por estado (definidos en jerarquico: TD_estado, LOG #45)
estado_familia, regla_estado = jq.estado_familia, jq.regla_estado


def linea_tiempo(con_ets=True):
    """Pronósticos de validación + puente + prueba (objetivos jul-2024 → sep-2026) con TD_* y, si existe, ETS."""
    partes = []
    for p in PERIODOS:
        d = pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet")
        d = d.merge(pd.read_parquet(SALIDA / f"pronosticos_{p}_jer.parquet"), on=["id_articulo", "origen", "mes_obj"])
        f = SALIDA / f"regimen_ets_{p}.parquet"
        if con_ets and f.exists():
            d = d.merge(pd.read_parquet(f), on=["id_articulo", "origen"])
        partes.append(d.assign(periodo=p))
    return pd.concat(partes, ignore_index=True).dropna(subset=["y"])


def etiquetar(d, est, racha, col="familia"):
    i = MESES.get_indexer(d.origen)
    j = est.columns.get_indexer(d[col])
    return d.assign(estado=est.to_numpy()[i, j], racha=racha.to_numpy()[i, j],
                    tramo=lambda x: pd.cut(x.racha, [0, 2, 5, 99], labels=["1–2 m", "3–5 m", "6+ m"]))


def sensibilidad(datos, d, bajos=(.75, .80, .85, .90, .95), altos=(1.10, 1.15, 1.20, 1.25)):
    """WAPE / sesgo / regla de adopción de A y B frente a ML_base para cada par de umbrales y periodo."""
    from metricas import adopta, por_origen, sesgo, wape
    filas = []
    for b in bajos:
        for a in altos:
            est, racha, _ = estado_familia(datos, b, a)
            x = etiquetar(d, est, racha)
            x["A"] = regla_estado(x, {"caída": "TD_mix_k3"})
            x["B"] = regla_estado(x)
            for p in ["validacion", "prueba"]:
                g = x[x.periodo == p]
                po = por_origen(g, ["ML_base", "A", "B"])
                w, s = wape(g, ["ML_base", "A", "B"]), sesgo(g, ["ML_base", "A", "B"])
                for m in ["A", "B"]:
                    filas.append({"bajo": b, "alto": a, "periodo": p, "regla": m, "WAPE": w[m],
                                  "dWAPE": w[m] - w.ML_base, "sesgo": s[m], "dsesgo_abs": abs(s[m]) - abs(s.ML_base),
                                  "gana": (po[m] < po.ML_base).mean(), "adopta": adopta(g, m, "ML_base"),
                                  "pct_caida": (g.estado == "caída").mean(), "pct_subida": (g.estado == "subida").mean()})
    # A no depende de `alto`: se deja una sola fila por umbral bajo
    s = pd.DataFrame(filas)
    return s[(s.regla == "B") | (s.alto == altos[0])].assign(alto=lambda x: x.alto.where(x.regla == "B"))


def main():
    t0 = time.time()
    datos = ft.Datos.cargar()
    for p in PERIODOS:
        td_ets(datos, pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet")).to_parquet(
            SALIDA / f"regimen_ets_{p}.parquet", index=False)
        print(f"ETS familia {p} ({time.time() - t0:.0f}s)")
    ablacion_regla(datos)
    print(f"ablación de la regla ({time.time() - t0:.0f}s)")
    sensibilidad(datos, linea_tiempo()).to_parquet(SALIDA / "regimen_sensibilidad.parquet", index=False)
    print(f"sensibilidad ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
