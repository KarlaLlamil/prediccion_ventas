"""Feature engineering (fase 2, DECISIONES.md §3.2 y §12; LOG_DECISIONES #16–#18).

Cada fila es un par (SKU, origen T) con objetivo y = unidades en T + H (H = 4). **Todo feature usa solo meses ≤ T**,
así que cualquier lag del objetivo es ≥ 4 (anti-leakage §3.2). La tabla de cada feature con su justificación está en
reports/FEATURES.md.

Pasos:
1. Matriz SKU × mes desde panel.parquet: NaN antes del nacimiento, 0 en meses sin venta (también después de fin_vida).
2. Limpieza **causal** de outliers (solo picos hacia arriba): tope = mediana de los 12 meses previos + 3.5 · IQR/1.349
   de los 24 previos, en log1p. Es causal (solo usa el pasado) para poder aplicarla igual en backtest y producción.
   Las caídas no se tocan: un destock es demanda real.
3. Estadísticos por mes para todo el panel (vectorizados) y se toma la columna del origen.

Uso:
    from features import Datos, construir
    datos = Datos.cargar()
    tabla = construir(datos, origenes)      # origenes: lista de índices de mes (0 = ene-2015)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"

H = 4
MESES = pd.date_range("2015-01-01", "2026-09-01", freq="MS")
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2022-12-01"))
DEV_REALES = ["Devolución de producto", "Garantía de Producto", "Cargos por devolucion (vtas)"]
MIN_HISTORIA = 13                     # meses desde el nacimiento para entrar al set (permite lag-12 y MASE estacional)


def _t(a):
    """Matriz SKU × mes → DataFrame mes × SKU (para usar rolling de pandas a lo largo del tiempo)."""
    return pd.DataFrame(a.T)


def _rolling(a, ventana, fn, min_periods=None):
    r = getattr(_t(a).rolling(ventana, min_periods=min_periods or ventana), fn)()
    return r.to_numpy().T


def _base_mediana(a):
    """Base local: mediana de los 12 meses previos (t−12…t−1); si es 0, la media (SKUs intermitentes)."""
    t = _t(a)
    med = t.rolling(12, min_periods=6).median().shift(1)
    media = t.rolling(12, min_periods=6).mean().shift(1)
    return med.where(med > 0, media).to_numpy().T


def _base_tendencia(a):
    """Base local con tendencia: recta OLS sobre los 12 meses previos proyectada a t (requiere 12 meses completos)."""
    w = sliding_window_view(a, 12, axis=1)                  # n × (T−11) × 12; la ventana k proyecta el mes k+12
    x = np.arange(12) - 5.5
    m = w.mean(axis=2)
    pend = ((w - m[..., None]) * x).sum(axis=2) / (x ** 2).sum()
    pred = np.clip(m + pend * 6.5, 0, None)
    b = np.full(a.shape, np.nan)
    b[:, 12:] = pred[:, :-1]
    return b


def _exceso(a, b, meses=6, tope=24):
    """Exceso acumulado: Σ de los últimos `meses` de (y − base), en meses de demanda "normal".
    El denominador es max(base, media de 12 meses) para que una base casi nula no dispare el cociente."""
    s = _t(a - b).rolling(meses, min_periods=meses).sum().to_numpy().T
    nivel = np.fmax(b, _rolling(a, 12, "mean"))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(np.where(nivel > 0, s / nivel, np.nan), -tope, tope)


def limpiar_outliers(y, z=3.5, piso=.25):
    """Tope causal de picos (paso 2 del docstring). Regresa (serie limpia, máscara de outliers)."""
    l = _t(np.log1p(y))
    base = l.rolling(12, min_periods=6).median().shift(1)
    q = l.rolling(24, min_periods=6)
    esc = ((q.quantile(.75) - q.quantile(.25)) / 1.349).shift(1).clip(lower=piso)
    tope = base + z * esc
    out = (l > tope).to_numpy().T
    lc = l.where(~(l > tope), tope)
    return np.expm1(lc.to_numpy().T), out


def indice_estacional(f):
    """Índice robusto por mes del año (EDA §4.4) con datos ≤ origen: y / mediana centrada de 13 meses, sin COVID,
    mediana entre años. f: DataFrame mes × serie. Un mes con < 3 años de dato → NaN (se usa el índice total)."""
    r = (f / f.rolling(13, center=True, min_periods=13).median()).replace([np.inf, -np.inf], np.nan)
    r[(r.index >= COVID[0]) & (r.index <= COVID[1])] = np.nan
    g = r.groupby(r.index.month)
    med = g.median().where(g.count() >= 3).reindex(range(1, 13))
    med = med.loc[:, med.notna().all()]                      # índice incompleto → se descarta la serie entera
    return (med / med.mean()).reindex(columns=f.columns)


@dataclass
class Datos:
    skus: pd.Index
    familia: np.ndarray
    i_nac: np.ndarray
    y: np.ndarray                     # unidades crudas (n_sku × n_mes)
    c: np.ndarray                     # unidades limpias (tope causal de picos)
    outlier: np.ndarray
    dev: np.ndarray                   # unidades devueltas (devolución + garantía + cargos), por mes de la orden
    perfil: np.ndarray

    @classmethod
    def cargar(cls):
        panel = pd.read_parquet(PROC / "panel.parquet")
        y = panel.pivot(index="id_articulo", columns="mes", values="unidades").reindex(columns=MESES)
        nac = panel.groupby("id_articulo").primera_venta.first().reindex(y.index)
        i_nac = MESES.get_indexer(nac)
        nacido = np.arange(len(MESES))[None, :] >= i_nac[:, None]
        y = y.where(~nacido | y.notna(), 0.0)
        fam = panel.groupby("id_articulo").familia.first().reindex(y.index).to_numpy()

        dev = pd.read_parquet(PROC / "devoluciones.parquet")
        dev = dev[dev.concepto.isin(DEV_REALES)]
        d = dev.groupby(["id_articulo", "mes"]).unidades_dev.sum().unstack().reindex(index=y.index, columns=MESES).fillna(0)

        perfil = pd.read_csv(RAIZ / "data" / "perfil_familia.csv", dtype=str).set_index("familia").perfil
        yv = y.to_numpy(dtype=float)
        c, out = limpiar_outliers(yv)
        return cls(y.index, fam, i_nac, yv, c, out, d.to_numpy(dtype=float), pd.Series(fam).map(perfil).fillna("mixto").to_numpy())


def _por_familia(datos, a):
    """Suma por familia de una matriz SKU × mes → (familias, matriz familia × mes)."""
    df = pd.DataFrame(np.nan_to_num(a), index=datos.familia)
    g = df.groupby(level=0).sum()
    return g.index.to_numpy(), g.to_numpy()


def _estadisticos(a):
    return {"ma3": _rolling(a, 3, "mean"), "ma6": _rolling(a, 6, "mean"), "ma12": _rolling(a, 12, "mean"),
            "sd6": _rolling(a, 6, "std"), "sd12": _rolling(a, 12, "std")}


def construir(datos, origenes):
    """Tabla de features para los orígenes dados (índices de mes). Incluye filas con objetivo desconocido
    (T + H > sep-2026) para poder pronosticar el futuro con la misma función."""
    y, c, n_mes = datos.y, datos.c, len(MESES)
    est = {"": _estadisticos(c), "_raw": _estadisticos(y)}

    # intermitencia acumulada desde el nacimiento (ADI y CV² de tamaños no nulos, como en el EDA)
    y0 = np.nan_to_num(y)
    nz = y0 > 0
    n_nz = np.cumsum(nz, axis=1)
    s1, s2 = np.cumsum(y0, axis=1), np.cumsum(y0 ** 2, axis=1)
    idx = np.where(nz, np.arange(n_mes)[None, :], -1)
    ultima = np.maximum.accumulate(idx, axis=1)
    venta12 = _rolling(nz.astype(float), 12, "sum")
    out12 = _rolling(datos.outlier.astype(float), 12, "sum")

    # proxy de sobre-stock (LOG #16d): sobre la serie CRUDA, porque el sobre-disparo es justo la señal
    sobre_cruda = _exceso(y0, _base_mediana(y0))
    sobre_tend = _exceso(y0, _base_tendencia(y0))
    previo = _t(y0).rolling(12, min_periods=12).mean().shift(3).to_numpy().T        # nivel de t−14…t−3
    with np.errstate(divide="ignore", invalid="ignore"):
        pulso3 = np.clip(np.where(previo > 0, _rolling(y0, 3, "mean") / previo - 1, np.nan), -1, 20)

    fams, fy = _por_familia(datos, y)
    _, fc = _por_familia(datos, c)
    pos_fam = pd.Index(fams).get_indexer(datos.familia)
    fam_sobre = _exceso(fy, _base_mediana(fy))
    tot = fy.sum(axis=0, keepdims=True)
    tot_sobre = _exceso(tot, _base_mediana(tot))[0]
    fam_ma3, fam_ma12 = _rolling(fc, 3, "mean"), _rolling(fc, 12, "mean")

    dev12 = _rolling(datos.dev, 12, "sum")
    vta12 = _rolling(y0, 12, "sum")
    _, fdev = _por_familia(datos, datos.dev)
    fdev12, fvta12 = _rolling(fdev, 12, "sum"), _rolling(fy, 12, "sum")

    fc_df = pd.DataFrame(fc.T, index=MESES, columns=fams)
    tot_df = pd.DataFrame(tot.T, index=MESES, columns=["total"])

    filas = []
    for o in origenes:
        vivo = (datos.i_nac <= o - (MIN_HISTORIA - 1)) & (venta12[:, o] > 0)
        k = np.flatnonzero(vivo)
        if not len(k):
            continue
        obj = o + H
        mes_obj = MESES[o] + pd.DateOffset(months=H)
        f = pos_fam[k]
        ie_f = indice_estacional(fc_df.iloc[:o + 1])
        ie_t = indice_estacional(tot_df.iloc[:o + 1])["total"]
        ie_tot = ie_t.get(mes_obj.month, np.nan)
        ie_fam = ie_f.loc[mes_obj.month].to_numpy()[f] if mes_obj.month in ie_f.index else np.full(len(k), np.nan)
        n_hist = o - datos.i_nac[k] + 1
        media_nz = s1[k, o] / np.maximum(n_nz[k, o], 1)
        var_nz = s2[k, o] / np.maximum(n_nz[k, o], 1) - media_nz ** 2

        r = {"id_articulo": datos.skus[k], "familia": datos.familia[k], "perfil": datos.perfil[k],
             "origen": MESES[o], "mes_obj": mes_obj,
             "y": y[k, obj] if obj < n_mes else np.nan}
        for suf, e in est.items():
            a = c if suf == "" else y
            esc = np.maximum(e["ma12"][k, o], 1e-3)
            r[f"escala{suf}"] = esc
            for nombre, col in [("lag4", o), ("lag5", o - 1), ("lag6", o - 2), ("lag12", obj - 12)]:
                r[f"{nombre}_n{suf}"] = a[k, col] / esc
            r[f"ma3_n{suf}"] = e["ma3"][k, o] / esc
            r[f"ma6_n{suf}"] = e["ma6"][k, o] / esc
            r[f"sd6_n{suf}"] = e["sd6"][k, o] / esc
            r[f"cv12{suf}"] = e["sd12"][k, o] / esc
        r.update({
            "log_escala": np.log(r["escala"]),
            "meses_sin_venta": o - ultima[k, o],
            "tasa_venta12": venta12[k, o] / 12,
            "adi": n_hist / np.maximum(n_nz[k, o], 1),
            "cv2": np.where(n_nz[k, o] > 1, var_nz / np.maximum(media_nz, 1e-9) ** 2, np.nan),
            "edad": n_hist,
            "outliers12": out12[k, o],
            "fam_momentum": fam_ma3[f, o] / np.maximum(fam_ma12[f, o], 1e-3),
            "fam_lag12_n": fc[f, obj - 12] / np.maximum(fam_ma12[f, o], 1e-3),
            "ie_fam": np.where(np.isnan(ie_fam), ie_tot, ie_fam),
            "ie_total": ie_tot,
            "cuota_fam": est[""]["ma12"][k, o] / np.maximum(fam_ma12[f, o], 1e-3),
            "mes_obj_n": mes_obj.month, "trimestre_obj": (mes_obj.month - 1) // 3 + 1,
            "fin_ano_obj": int(mes_obj.month in (11, 12)), "anio_obj": mes_obj.year,
            "sobre_cruda": sobre_cruda[k, o], "sobre_tend": sobre_tend[k, o], "pulso3": pulso3[k, o],
            "sobre_fam": fam_sobre[f, o], "sobre_total": tot_sobre[o],
            "tasa_dev12": np.clip(dev12[k, o] / np.maximum(vta12[k, o], 1), 0, 2),
            "tasa_dev12_fam": np.clip(fdev12[f, o] / np.maximum(fvta12[f, o], 1), 0, 2),
        })
        filas.append(pd.DataFrame(r))
    return pd.concat(filas, ignore_index=True)


# Grupos de features para el modelo global y las ablaciones (LOG #16d, §10 experimento de devoluciones)
BASE = ["lag4_n", "lag5_n", "lag6_n", "lag12_n", "ma3_n", "ma6_n", "sd6_n", "cv12", "log_escala",
        "meses_sin_venta", "tasa_venta12", "adi", "cv2", "edad", "outliers12",
        "fam_momentum", "fam_lag12_n", "ie_fam", "ie_total", "cuota_fam",
        "mes_obj_n", "trimestre_obj", "fin_ano_obj", "anio_obj", "familia"]
SIN_LIMPIEZA = {f: f"{f}_raw" for f in ["lag4_n", "lag5_n", "lag6_n", "lag12_n", "ma3_n", "ma6_n", "sd6_n", "cv12"]}
SOBRE_SKU = ["sobre_cruda", "sobre_tend", "pulso3"]
SOBRE_AGR = ["sobre_fam", "sobre_total"]
DEVOL = ["tasa_dev12", "tasa_dev12_fam"]
