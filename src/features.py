"""Feature engineering (fase 2, DECISIONES.md §3.2 y §12; LOG_DECISIONES #16–#18).

Cada fila es un par (SKU, origen T) con objetivo y = unidades en T + H (H = 3). **Todo feature usa solo meses ≤ T**,
así que cualquier lag del objetivo es ≥ H (anti-leakage §3.2; `construir` lo verifica con un assert). Los lags del
objetivo se nombran por su distancia al mes objetivo: lag3 = T, lag4 = T−1, lag5 = T−2, lag12 = T+H−12. La tabla de
cada feature con su justificación está en reports/FEATURES.md.

Pasos:
1. Matriz SKU × mes desde panel.parquet: NaN antes del nacimiento, 0 en meses sin venta (también después de fin_vida).
2. Limpieza **causal** (cada mes t solo usa meses < t, así que vale igual en backtest y producción):
   a. Meses anormalmente bajos del **total**: log1p(total) < mediana de los 12 meses previos − 3.5 · IQR/1.349 de los
      24 previos (piso de escala 0.1). En esos meses la serie limpia de cada SKU se sube al piso del total
      (× piso / total). Las rachas se nombran `caida_AAAA_MM[_MM]`; como mínimo cubre caida_2020_04_05. Esos meses
      tampoco se usan como objetivo de entrenamiento (`Datos.bajo`).
   b. Picos por SKU: tope = mediana de los 12 meses previos + 3.5 · IQR/1.349 de los 24 previos, en log1p.
3. La limpieza solo se **aplica** a los SKUs suaves y erráticos según su clase Syntetos–Boylan al origen
   (`Datos.serie(o)`); en grumosos e intermitentes los picos son la demanda. Los lags, medias, el objetivo de
   entrenamiento (`y_entreno`) y los agregados de familia usan esa serie; la evaluación siempre es contra `y` cruda.
4. Estadísticos por mes para todo el panel (vectorizados) y se toma la columna del origen.

Clase SB (causal, igual que en el EDA): ADI = meses desde el nacimiento hasta la **última venta** ≤ T / meses con
venta (sin contar los ceros posteriores a la última venta); CV² = (sd / media)² de los tamaños no nulos.

Uso:
    from features import Datos, construir
    datos = Datos.cargar()
    tabla = construir(datos, origenes)      # origenes: lista de índices de mes (0 = ene-2015)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metricas import clase_sb

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "data" / "processed"

H = 3
MESES = pd.date_range("2015-01-01", "2026-09-01", freq="MS")
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2022-12-01"))
DEV_REALES = ["Devolución de producto", "Garantía de Producto", "Cargos por devolucion (vtas)"]
MIN_HISTORIA = 13                     # meses desde el nacimiento para entrar al set (permite lag-12 y MASE estacional)
LAGS = {f"lag{l}": l for l in (H, H + 1, H + 2, 12)}   # nombre → distancia (meses) del lag al mes objetivo
CLASES_LIMPIAS = ("suave", "errática")                 # clases SB a las que se aplica la limpieza
Z_OUT, Z_BAJO, PISO_BAJO = 3.5, 3.5, .1


def _t(a):
    """Matriz SKU × mes → DataFrame mes × SKU (para usar rolling de pandas a lo largo del tiempo)."""
    return pd.DataFrame(a.T)


def _rolling(a, ventana, fn, min_periods=None):
    r = getattr(_t(a).rolling(ventana, min_periods=min_periods or ventana), fn)()
    return r.to_numpy().T


def _tope(l, z, piso, signo):
    """Umbral causal en log: mediana de los 12 previos ± z · IQR/1.349 de los 24 previos (escala con piso)."""
    base = l.rolling(12, min_periods=6).median().shift(1)
    q = l.rolling(24, min_periods=6)
    esc = ((q.quantile(.75) - q.quantile(.25)) / 1.349).shift(1).clip(lower=piso)
    return base + signo * z * esc


def limpiar_outliers(y, z=Z_OUT, piso=.25):
    """Tope causal de picos (paso 2b del docstring). Regresa (serie limpia, máscara de outliers)."""
    l = _t(np.log1p(y))
    tope = _tope(l, z, piso, +1)
    out = (l > tope).to_numpy().T
    lc = l.where(~(l > tope), tope)
    return np.expm1(lc.to_numpy().T), out


def meses_bajos(tot, z=Z_BAJO, piso=PISO_BAJO):
    """Meses anormalmente bajos del total (paso 2a). Regresa (máscara por mes, factor para subir al piso)."""
    l = pd.Series(np.log1p(tot))
    p = _tope(l, z, piso, -1)
    bajo = (l < p).to_numpy()
    factor = np.where(bajo, np.expm1(p.to_numpy()) / np.maximum(tot, 1), 1.0)
    return bajo, factor


def nombres_bajos(bajo):
    """Rachas de meses bajos → nombres `caida_AAAA_MM[_MM]` (p. ej. caida_2020_04_05)."""
    i = np.flatnonzero(bajo)
    rachas = np.split(i, np.flatnonzero(np.diff(i) > 1) + 1) if len(i) else []
    return {f"caida_{MESES[r[0]]:%Y_%m}" + "".join(f"_{MESES[j]:%m}" for j in r[1:]): list(MESES[r]) for r in rachas}


def clase_sb_causal(y, i_nac):
    """Clase SB por SKU × mes con datos ≤ t (ADI hasta la última venta; ver docstring). "" antes de la 1.ª venta."""
    y0 = np.nan_to_num(y)
    nz = y0 > 0
    n_nz = np.cumsum(nz, axis=1)
    s1, s2 = np.cumsum(y0, axis=1), np.cumsum(y0 ** 2, axis=1)
    ultima = np.maximum.accumulate(np.where(nz, np.arange(y.shape[1])[None, :], -1), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        adi = (ultima - i_nac[:, None] + 1) / np.maximum(n_nz, 1)
        media = s1 / np.maximum(n_nz, 1)
        cv2 = np.where(n_nz > 1, (s2 / np.maximum(n_nz, 1) - media ** 2) / np.maximum(media, 1e-9) ** 2, np.nan)
    clase = clase_sb(adi.ravel(), cv2.ravel()).reshape(y.shape).astype(object)
    clase[n_nz == 0] = ""
    return clase, adi, cv2, n_nz, ultima


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
    c: np.ndarray                     # unidades limpias (meses bajos del total al piso + tope causal de picos)
    outlier: np.ndarray
    dev: np.ndarray                   # unidades devueltas (devolución + garantía + cargos), por mes de la orden
    perfil: np.ndarray
    clase: np.ndarray                 # clase SB causal SKU × mes
    bajo: np.ndarray                  # meses bajos del total (por mes)

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
        bajo, factor = meses_bajos(np.nansum(yv, axis=0))
        assert bajo[MESES.get_indexer([pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01")])].all(), \
            "la detección de meses bajos no cubre caida_2020_04_05"
        c, out = limpiar_outliers(yv * factor[None, :])
        clase = clase_sb_causal(yv, i_nac)[0]
        return cls(y.index, fam, i_nac, yv, c, out, d.to_numpy(dtype=float),
                   pd.Series(fam).map(perfil).fillna("mixto").to_numpy(), clase, bajo)

    def limpiable(self, o):
        """SKUs a los que se aplica la limpieza en el origen o (clase SB al origen suave o errática)."""
        return np.isin(self.clase[:, o], CLASES_LIMPIAS)

    def serie(self, o):
        """Serie de modelado vista desde el origen o: limpia para suaves/erráticos, cruda para el resto."""
        return np.where(self.limpiable(o)[:, None], self.c, self.y)

    @property
    def meses_bajos(self):
        return MESES[self.bajo]


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
    corto = {n: l for n, l in LAGS.items() if l < H}
    assert not corto, f"lags menores que el horizonte H = {H} (leakage): {corto}"
    y, c, n_mes = datos.y, datos.c, len(MESES)
    e_c, e_y = _estadisticos(c), _estadisticos(y)

    # intermitencia acumulada desde el nacimiento (ADI hasta la última venta y CV² de tamaños no nulos, como el EDA)
    _, adi, cv2, n_nz, ultima = clase_sb_causal(y, datos.i_nac)
    nz = np.nan_to_num(y) > 0
    venta12 = _rolling(nz.astype(float), 12, "sum")
    out12 = _rolling(datos.outlier.astype(float), 12, "sum")
    y0 = np.nan_to_num(y)

    dev12 = _rolling(datos.dev, 12, "sum")
    vta12 = _rolling(y0, 12, "sum")
    fams, fy = _por_familia(datos, y)
    pos_fam = pd.Index(fams).get_indexer(datos.familia)
    _, fdev = _por_familia(datos, datos.dev)
    fdev12, fvta12 = _rolling(fdev, 12, "sum"), _rolling(fy, 12, "sum")

    filas = []
    for o in origenes:
        vivo = (datos.i_nac <= o - (MIN_HISTORIA - 1)) & (venta12[:, o] > 0)
        k = np.flatnonzero(vivo)
        if not len(k):
            continue
        obj = o + H
        mes_obj = MESES[o] + pd.DateOffset(months=H)
        f = pos_fam[k]
        limpia = datos.limpiable(o)
        lk = limpia[k]
        sel = lambda a, b: np.where(lk, a, b)                # limpio para suaves/erráticos, crudo para el resto

        # agregados de familia y total con la serie de modelado vista desde el origen (solo meses ≤ o)
        _, fc = _por_familia(datos, datos.serie(o)[:, :o + 1])
        fam_ma3, fam_ma12 = fc[:, -3:].mean(axis=1), fc[:, -12:].mean(axis=1)
        fc_df = pd.DataFrame(fc.T, index=MESES[:o + 1], columns=fams)
        ie_f = indice_estacional(fc_df)
        ie_t = indice_estacional(fc_df.sum(axis=1).to_frame("total"))["total"]
        ie_tot = ie_t.get(mes_obj.month, np.nan)
        ie_fam = ie_f.loc[mes_obj.month].to_numpy()[f] if mes_obj.month in ie_f.index else np.full(len(k), np.nan)

        r = {"id_articulo": datos.skus[k], "familia": datos.familia[k], "perfil": datos.perfil[k],
             "origen": MESES[o], "mes_obj": mes_obj,
             "y": y[k, obj] if obj < n_mes else np.nan,
             "y_entreno": sel(c[k, obj], y[k, obj]) if obj < n_mes else np.nan,
             "obj_bajo": bool(datos.bajo[obj]) if obj < n_mes else False,
             "limpia": lk}
        for suf, a, e in [("", None, None), ("_raw", y, e_y)]:
            if suf == "":                                    # serie de modelado: limpia o cruda según la clase al origen
                lag = lambda l: sel(c[k, obj - l], y[k, obj - l])
                est = {s: sel(e_c[s][k, o], e_y[s][k, o]) for s in e_c}
            else:
                lag = lambda l: y[k, obj - l]
                est = {s: e[s][k, o] for s in e}
            esc = np.maximum(est["ma12"], 1e-3)
            r[f"escala{suf}"] = esc
            for nombre, l in LAGS.items():
                r[f"{nombre}_n{suf}"] = lag(l) / esc
            r[f"ma3_n{suf}"] = est["ma3"] / esc
            r[f"ma6_n{suf}"] = est["ma6"] / esc
            r[f"sd6_n{suf}"] = est["sd6"] / esc
            r[f"cv12{suf}"] = est["sd12"] / esc
        r.update({
            "log_escala": np.log(r["escala"]),
            "meses_sin_venta": o - ultima[k, o],
            "tasa_venta12": venta12[k, o] / 12,
            "adi": adi[k, o],
            "cv2": cv2[k, o],
            "edad": o - datos.i_nac[k] + 1,
            "outliers12": out12[k, o],
            "fam_momentum": fam_ma3[f] / np.maximum(fam_ma12[f], 1e-3),
            "fam_lag12_n": fc[f, obj - 12] / np.maximum(fam_ma12[f], 1e-3),
            "ie_fam": np.where(np.isnan(ie_fam), ie_tot, ie_fam),
            "ie_total": ie_tot,
            "cuota_fam": r["escala"] / np.maximum(fam_ma12[f], 1e-3),
            "mes_obj_n": mes_obj.month, "trimestre_obj": (mes_obj.month - 1) // 3 + 1,
            "fin_ano_obj": int(mes_obj.month in (11, 12)), "anio_obj": mes_obj.year,
            "tasa_dev12": np.clip(dev12[k, o] / np.maximum(vta12[k, o], 1), 0, 2),
            "tasa_dev12_fam": np.clip(fdev12[f, o] / np.maximum(fvta12[f, o], 1), 0, 2),
        })
        filas.append(pd.DataFrame(r))
    return pd.concat(filas, ignore_index=True)


# Grupos de features para el modelo global y las ablaciones (LOG #16d, §10 experimento de devoluciones)
LAG_FEATS = [f"{n}_n" for n in LAGS]                  # lag3_n, lag4_n, lag5_n, lag12_n
BASE = LAG_FEATS + ["ma3_n", "ma6_n", "sd6_n", "cv12", "log_escala",
                    "meses_sin_venta", "tasa_venta12", "adi", "cv2", "edad", "outliers12",
                    "fam_momentum", "fam_lag12_n", "ie_fam", "ie_total", "cuota_fam",
                    "mes_obj_n", "trimestre_obj", "fin_ano_obj", "anio_obj", "familia"]
SIN_LIMPIEZA = {f: f"{f}_raw" for f in LAG_FEATS + ["ma3_n", "ma6_n", "sd6_n", "cv12"]}
DEVOL = ["tasa_dev12", "tasa_dev12_fam"]
