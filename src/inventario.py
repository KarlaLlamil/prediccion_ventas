"""Recomendación de pedido y simulación de inventario (fase 4; DECISIONES.md §14, LOG_DECISIONES #25).

Política order-up-to con revisión mensual (R = 1); la ventana de protección R + L es el horizonte H = 3 (L = H − R = 2):
    al cierre del mes T se pide  Q_T = max(0, S_T − posición),  S_T = pronóstico de la ventana T+1…T+H + SS_T
    posición = on-hand tras la recepción del cierre de T + tránsito (pedidos de T−H+2 … T−1; con H = 3, el de T−1)
    Q_T llega al cierre de T+H−1 y surte T+H.
Pronóstico de la ventana = suma de los pronósticos h = H ya emitidos para cada mes (T+1 se pronosticó en T+1−H, …,
T+H en T). Todos usan datos ≤ T, así que no hay leakage. Si falta un mes (SKU que no estaba en el set en ese
origen), se usa el pronóstico de T+H como proxy plano.

Pronóstico por SKU y mes: el modelo evaluado para SKUs maduros, `Media_post` para los jóvenes (LOG #24) y 0 para
los dormidos (sin venta en 12 meses): la política deja de pedirles y su stock se reporta como exposición. Se
conservan como repuesto y pueden reactivarse; su venta de reactivación se reporta aparte (`cobertura.csv`).

Stock de seguridad (SS), con errores observados ≤ T:
    sin_ss     SS = 0
    normal     z_α · √H · RMSE del SKU (errores mensuales de los últimos 12 meses; √H meses de ventana, supuesto de
               independencia). Con < 6 errores se usa el RMSE relativo de su clase × escala.
    empirico   cuantil α del error de ventana relativo (Σ H meses de (real − pronóstico) / escala), agrupado por clase
               SB (o "joven") × escala del SKU. No supone normalidad y absorbe el sesgo del modelo (§14: los
               intermitentes no usan la fórmula normal).
    Arranque (< MIN_POOL errores de ventana): SS = 1 mes de pronóstico.

Supuestos (§14): ventas perdidas (lo no surtido no se recupera), sin órdenes abiertas al inicio, arranque ideal de
cada SKU (on-hand = 1 mes + SS, tránsito = pronóstico). Por eso las métricas se miden después del calentamiento:
validación ene–jun 2025 (demanda ≤ primer origen de prueba, para que la política no se elija con datos posteriores)
y prueba sep-2025 → sep-2026. Jul–ago 2025 usan los pronósticos del periodo puente del backtest (solo continuidad).

Uso:  venv\\Scripts\\python.exe src\\inventario.py          (escribe data/processed/resultados/inv_*.csv)
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

import features as ft
from backtest import SALIDA
from features import H, MESES, PROC
from jerarquico import modelo_elegido

NIVELES = [.80, .90, .95, .98]
NIVEL_ABC = {"A": .95, "B": .90, "C": .80, "sin_ABC": .90}   # sin_ABC = SKUs nuevos después de ago-2025
MODELOS = [modelo_elegido(), "ML_base", "Ruta_clasica", "MA12", "SNaive"]   # TD_estado_LGBM_tw = modelo elegido (LOG #50)
METODOS = ["sin_ss", "normal", "empirico"]
MIN_POOL = 200
MIN_ERR_SKU = 6
I = MESES.get_loc
M0, M1 = I(pd.Timestamp("2024-09-01")), I(pd.Timestamp("2026-09-01"))      # meses objetivo con pronóstico
PERIODOS = {"validacion": ("2025-01-01", "2025-06-01"), "prueba": ("2025-09-01", "2026-09-01")}
FUENTES = ("validacion", "puente", "prueba")                                 # periodos del backtest que se leen


class Insumos:
    """Matrices SKU × mes (índice de MESES) de demanda real, pronóstico h = H, escala y clase en el origen."""

    def __init__(self, modelo, datos=None, extra=None):
        """`extra`: DataFrame (id_articulo, origen, modelo) con un pronóstico que no está en los parquet del backtest
        (p. ej. `TD_estado_LGBM_tw`, `inventario_lgbm.py`); debe cubrir validación, puente y prueba."""
        datos = datos or ft.Datos.cargar()
        mad = pd.concat([pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet") for p in FUENTES])
        if extra is None and modelo == "TD_estado_LGBM_tw":    # modelo adoptado (LOG #50), de `inventario_lgbm.py`
            extra = pd.read_parquet(SALIDA / "td_lgbm_tw_inventario.parquet")
        if extra is not None:
            mad = mad.merge(extra[["id_articulo", "origen", modelo]], on=["id_articulo", "origen"], how="left")
            assert mad[modelo].notna().all(), f"{modelo}: faltan pronósticos en `extra`"
        elif modelo.startswith("TD_"):                         # pronóstico jerárquico (src/jerarquico.py, LOG #32)
            jer = pd.concat([pd.read_parquet(SALIDA / f"pronosticos_{p}_jer.parquet") for p in FUENTES])
            mad = mad.merge(jer[["id_articulo", "origen", modelo]], on=["id_articulo", "origen"], how="left")
        esc = pd.read_parquet(PROC / "features.parquet", columns=["id_articulo", "origen", "escala"])
        mad = mad.merge(esc, on=["id_articulo", "origen"], how="left")
        jov = pd.concat([pd.read_parquet(SALIDA / f"cold_jovenes_{p}.parquet") for p in FUENTES])
        filas = pd.concat([mad[["id_articulo", "mes_obj", modelo, "escala", "clase"]].rename(columns={modelo: "f"}),
                           jov[["id_articulo", "mes_obj", "Media_post", "escala"]].rename(columns={"Media_post": "f"})
                           .assign(clase="joven")], ignore_index=True)
        self.skus = pd.Index(sorted(filas.id_articulo.unique()))
        n, T = len(self.skus), len(MESES)
        r = self.skus.get_indexer(filas.id_articulo)
        c = MESES.get_indexer(filas.mes_obj)
        self.F = np.zeros((n, T))
        self.esc = np.full((n, T), np.nan)
        self.clase = np.full((n, T), "", dtype=object)
        self.tiene = np.zeros((n, T), bool)                 # hay pronóstico emitido en el origen m − H
        self.F[r, c], self.esc[r, c], self.clase[r, c], self.tiene[r, c] = filas.f, filas.escala, filas.clase, True
        self.D = np.nan_to_num(datos.y[datos.skus.get_indexer(self.skus)])
        self.familia = datos.familia[datos.skus.get_indexer(self.skus)]
        self.inicio = np.argmax(self.tiene, axis=1) - H      # primer origen con pronóstico (decisión de arranque)
        precio = pd.read_parquet(PROC / "precio_sku_mes.parquet")
        precio = precio[precio.mes >= MESES[M0]].groupby("id_articulo").precio_mxn.median()
        self.precio = precio.reindex(self.skus).fillna(precio.median()).to_numpy()
        abc = pd.read_parquet(PROC / "clasificacion_sku.parquet").set_index("id_articulo").abc
        self.abc = abc.reindex(self.skus).astype(object).fillna("sin_ABC").to_numpy()


def _ventana(ins, T):
    """Pronóstico de T+1…T+H con lo emitido hasta T (proxy plano del de T+H donde falta)."""
    m = np.arange(T + 1, T + H + 1)
    f = np.where(ins.tiene[:, m], ins.F[:, m], ins.F[:, [T + H]])
    return f.sum(axis=1), f


def _errores_ventana(ins, T):
    """Errores de ventana relativos ya observados en T: decisiones T' con T' + H ≤ T y los H meses pronosticados."""
    filas = []
    for Tp in range(M0 - 1, T - H + 1):
        m = np.arange(Tp + 1, Tp + H + 1)
        ok = ins.tiene[:, m].all(axis=1) & (ins.esc[:, Tp + H] > 0)
        if ok.any():
            e = (ins.D[ok][:, m] - ins.F[ok][:, m]).sum(axis=1) / ins.esc[ok, Tp + H]
            filas.append(pd.DataFrame({"clase": ins.clase[ok, Tp + H], "e": e}))
    return pd.concat(filas) if filas else pd.DataFrame(columns=["clase", "e"])


def ss_empirico(err, clase, esc, alfa):
    """Cuantil α del error de ventana relativo de la clase (o de todas, si la clase tiene < MIN_POOL) × escala."""
    g = err.groupby("clase").e
    q = g.quantile(alfa).where(g.size() >= MIN_POOL)
    qc = pd.Series(clase).map(q).fillna(err.e.quantile(alfa)).to_numpy()
    return np.maximum(np.nan_to_num(qc * esc), 0)


def stock_seguridad(ins, T, metodo, alfa, f_ventana):
    """SS por SKU. `alfa` puede ser un número, un dict ABC → nivel (servicio diferenciado con el ABC estático) o una
    matriz SKU × mes de niveles (segmentación dinámica, `segmentacion.py`; nivel 0 = sin SS)."""
    if isinstance(alfa, np.ndarray):
        nivel = alfa[:, T]
    elif isinstance(alfa, dict):
        nivel = pd.Series(ins.abc).map(alfa).to_numpy()
    else:
        return _ss(ins, T, metodo, alfa, f_ventana)
    ss = np.zeros(len(ins.skus))
    for a in np.unique(nivel):
        if a > 0:
            ss[nivel == a] = _ss(ins, T, metodo, a, f_ventana)[nivel == a]
    return ss


def _ss(ins, T, metodo, alfa, f_ventana):
    n = len(ins.skus)
    esc, clase = ins.esc[:, T + H], ins.clase[:, T + H]
    vivo = ins.tiene[:, T + H] & (esc > 0)
    if metodo == "sin_ss":
        return np.zeros(n)
    if metodo == "empirico":
        err = _errores_ventana(ins, T)
        if len(err) < MIN_POOL:
            return np.where(vivo, f_ventana[:, -1], 0.0)
        return np.where(vivo, ss_empirico(err, clase, esc, alfa), 0.0)
    # normal
    m = np.arange(max(M0, T - 11), T + 1)
    obs = ins.tiene[:, m]
    if obs.sum() < MIN_POOL:
        return np.where(vivo, f_ventana[:, -1], 0.0)
    e = np.where(obs, ins.D[:, m] - ins.F[:, m], np.nan)
    n_obs = obs.sum(axis=1)
    rmse = np.sqrt(np.nansum(e ** 2, axis=1) / np.maximum(n_obs, 1))
    rel = pd.DataFrame({"clase": ins.clase[:, m].ravel(), "r": (e / ins.esc[:, m]).ravel()}).dropna()
    rmse_rel = rel.groupby("clase").r.apply(lambda s: np.sqrt((s ** 2).mean()))
    rmse_pool = pd.Series(clase).map(rmse_rel).fillna(np.sqrt((rel.r ** 2).mean())).to_numpy() * esc
    sigma = np.where(n_obs >= MIN_ERR_SKU, rmse, rmse_pool)
    return np.where(vivo, norm.ppf(alfa) * np.sqrt(H) * np.nan_to_num(sigma), 0.0)


def simular(ins, metodo, alfa):
    """Corre la política mes a mes (vectorizada sobre SKUs). Regresa matrices SKU × mes de demanda atendible,
    venta surtida, stock promedio y pedidos."""
    n, T_ = len(ins.skus), len(MESES)
    oh = np.zeros(n)                                     # on-hand al inicio del mes t (tras recepción)
    arr = np.zeros((n, T_ + H + 1))                      # arr[:, m]: llega para surtir el mes m
    activo = np.zeros(n, bool)
    dem, ven, inv, ped = (np.zeros((n, T_)) for _ in range(4))
    for t in range(M0 - 1, M1 + 1):
        if t >= M0:                                      # 1) demanda del mes t
            a = activo
            dem[a, t] = ins.D[a, t]
            ven[a, t] = np.minimum(oh[a], ins.D[a, t])
            inv[a, t] = oh[a] - ven[a, t] / 2            # stock promedio del mes
            oh = oh - ven[:, t]
            oh = oh + arr[:, t + 1]                      # 2) recepción al cierre de t
        if t + H > M1:
            continue
        fv, f = _ventana(ins, t)                         # 3) decisión al cierre de t
        ss = stock_seguridad(ins, t, metodo, alfa, f)
        nuevo = (ins.inicio <= t) & ~activo
        if nuevo.any():                                  # arranque ideal del SKU
            oh[nuevo] = f[nuevo, 0] + ss[nuevo]
            for j in range(2, H):                        # tránsito = pronóstico de T+2 … T+H−1
                arr[nuevo, t + j] = f[nuevo, j - 1]
            activo |= nuevo
        posicion = oh + arr[:, t + 2:t + H].sum(axis=1)  # pedidos en tránsito (llegan para surtir T+2 … T+H−1)
        q = np.where(activo, np.maximum(fv + ss - posicion, 0), 0.0)
        arr[:, t + H] = q
        ped[:, t] = q
    return dem, ven, inv, ped


def metricas(ins, sim, desde, hasta, filtro=None):
    dem, ven, inv, ped = sim
    m = np.arange(I(pd.Timestamp(desde)), I(pd.Timestamp(hasta)) + 1)
    k = np.ones(len(ins.skus), bool) if filtro is None else filtro
    d, v, s = dem[k][:, m], ven[k][:, m], inv[k][:, m]
    p = ins.precio[k][:, None]
    con_dem = d > 0
    return {"fill_rate": v.sum() / d.sum(),
            "fill_rate_mxn": (v * p).sum() / (d * p).sum(),
            "pct_meses_quiebre": ((v < d) & con_dem).sum() / con_dem.sum(),
            "stock_prom_u": s.sum(axis=0).mean(),
            "stock_prom_mxn": (s * p).sum(axis=0).mean(),
            "cobertura_meses": (s * p).sum() / (d * p).sum(),
            "venta_perdida_mxn": ((d - v) * p).sum(axis=0).mean()}


def stock_en_riesgo(ins, sim):
    """Valor del stock al cierre de sep-2026 en SKUs sin venta real en los últimos 6 meses. No es obsolescencia: parte
    se conserva a propósito como repuesto (SKUs dormidos que pueden volver a venderse); se reporta como exposición."""
    dem, ven, inv, ped = sim
    final = inv[:, M1] - ven[:, M1] / 2
    sin_venta = ins.D[:, M1 - 5:M1 + 1].sum(axis=1) == 0
    return float((final * ins.precio)[sin_venta].sum())


def main():
    t0 = time.time()
    datos = ft.Datos.cargar()
    filas, seg = [], []
    for modelo in MODELOS:
        ins = Insumos(modelo, datos)
        abc = ins.abc
        politicas = [("sin_ss", 0)] + [(m, a) for m in METODOS[1:] for a in NIVELES] + [("empirico", NIVEL_ABC)]
        for metodo, alfa in politicas:
            sim = simular(ins, metodo, alfa)
            base = {"modelo": modelo, "metodo": metodo,
                    "nivel": "ABC 95/90/80" if isinstance(alfa, dict) else (alfa or None)}
            if True:
                for periodo, (a, b) in PERIODOS.items():
                    filas.append({**base, "periodo": periodo, **metricas(ins, sim, a, b),
                                  "stock_riesgo_mxn": stock_en_riesgo(ins, sim) if periodo == "prueba" else np.nan})
                    for g in np.unique(abc):
                        seg.append({**base, "periodo": periodo, "abc": g, **metricas(ins, sim, a, b, abc == g)})
        print(f"  {modelo} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(filas)
    res.to_csv(SALIDA / "inv_simulacion.csv", index=False)
    pd.DataFrame(seg).to_csv(SALIDA / "inv_simulacion_abc.csv", index=False)
    print(res[res.periodo == "prueba"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
