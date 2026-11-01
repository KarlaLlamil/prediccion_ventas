"""Segmentación dinámica por origen (fase 4b, LOG #31): ABC móvil, fase de ciclo de vida y servicio diferenciado.

ABC móvil: en cada mes t, solo con datos ≤ t.
    valor 12 m = Σ unidades × precio MXN del mes, meses t−11…t
    SKUs con < 12 meses de vida → valor anualizado sin el mes de lanzamiento (llenado de canal, LOG #24):
        edad 1–11:  (valor 12 m − valor del mes 0) × 12 / edad
        edad 0:     valor del mes 0 × 12 / FACTOR_LANZ (mes 0 ÷ media mensual de los meses 1–11, lanzamientos ≤ ago-2024)
    A / B / C = hasta 80 / 95 / 100 % del valor acumulado entre los SKUs con valor > 0. Sin venta en 12 m → C.

Fase de ciclo de vida en t:
    introducción   edad < 12 meses
    dormido        edad ≥ 12 y sin venta en los últimos 12 meses (N_DORMIDO de preparar.py): repuesto que se conserva
                   y puede volver a venderse; sin SS (el stock que tenga es remanente de repuesto, no protección)
    crecimiento / madurez / declive: r = media de los últimos 6 meses ÷ media base, donde la base son los mismos
                   6 meses del año anterior (quita la estacionalidad), sin el mes de lanzamiento. Si no hay base
                   (SKU de 12 meses), se usan los meses posteriores al lanzamiento previos a los últimos 6.
                   r ≥ 1.3 → crecimiento, r ≤ 1/1.3 → declive, en medio → madurez. Sin venta en 6 meses (pero sí
                   en 12) → declive.

Servicio diferenciado: se construye una matriz SKU × mes de niveles α que `inventario.stock_seguridad` lee en cada
decisión (el nivel de T solo usa datos ≤ T). Los ajustes por fase mueven el nivel por la ESCALERA de niveles.

Uso:  venv\\Scripts\\python.exe src\\segmentacion.py   (escribe data/processed/segmentacion_sku_mes.parquet e
      data/processed/resultados/inv_segmentacion*.csv)
"""
import time

import numpy as np
import pandas as pd

import features as ft
import inventario as iv
from backtest import SALIDA
from features import MESES, PROC
from preparar import N_DORMIDO

FIN_FACTOR = MESES.get_loc(pd.Timestamp("2024-08-01"))   # último mes de datos para calibrar FACTOR_LANZ (antes de M0)
UMBRAL_TEND = 1.3
FASES = ["introducción", "crecimiento", "madurez", "declive", "dormido"]
ESCALERA = [0.0, .70, .80, .90, .95, .98]                # niveles posibles; 0 = sin SS
ABC_959080 = {"A": .95, "B": .90, "C": .80}
AJUSTE_FASE = {"introducción": -1, "crecimiento": +1, "madurez": 0, "declive": -1, "dormido": None}  # None → sin SS
UNIFORMES = [.70, .80, .90, .95, .98]


def _suma_movil(a, k):
    """Σ de los últimos k meses (incluido t) por fila; al inicio suma lo que haya."""
    c = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)
    t = np.arange(a.shape[1])
    return c[:, t + 1] - c[:, np.maximum(t - k + 1, 0)]


def _rezago(a, k):
    out = np.zeros_like(a)
    out[:, k:] = a[:, :-k]
    return out


def matriz_precio(skus):
    p = pd.read_parquet(PROC / "precio_sku_mes.parquet").pivot(index="id_articulo", columns="mes", values="precio_mxn")
    p = p.reindex(index=skus, columns=MESES).ffill(axis=1).bfill(axis=1)
    return p.fillna(np.nanmedian(p.to_numpy())).to_numpy()


def edad_meses(datos):
    i_nac = np.where(datos.i_nac >= 0, datos.i_nac, -10 ** 4)
    return np.arange(len(MESES))[None, :] - i_nac[:, None], i_nac


def factor_lanzamiento(datos):
    """Mes de lanzamiento ÷ media mensual de los meses 1–11 (Σ sobre lanzamientos completos ≤ FIN_FACTOR)."""
    y = np.nan_to_num(datos.y)
    _, i_nac = edad_meses(datos)
    ok = np.where((i_nac >= 0) & (i_nac + 11 <= FIN_FACTOR))[0]
    m0 = y[ok, i_nac[ok]].sum()
    resto = sum(y[i, i_nac[i] + 1:i_nac[i] + 12].mean() for i in ok)
    return m0 / resto


def abc_movil(datos, precio, factor, anualizar=True):
    """(valor anual, clase ABC) SKU × mes. Clase "" = SKU aún no lanzado."""
    y = np.nan_to_num(datos.y)
    n, T = y.shape
    edad, i_nac = edad_meses(datos)
    v = y * precio
    anual = _suma_movil(v, 12)
    if anualizar:
        v0 = np.where(i_nac >= 0, v[np.arange(n), np.maximum(i_nac, 0)], 0)[:, None]
        joven = (edad >= 1) & (edad < 12)
        anual = np.where(joven, (anual - v0) * 12 / np.maximum(edad, 1), anual)
        anual = np.where(edad == 0, v0 * 12 / factor, anual)
    anual = np.where(edad >= 0, np.maximum(anual, 0), 0)
    clase = np.full((n, T), "", dtype=object)
    for t in range(T):
        a = anual[:, t]
        pos = np.where(a > 0)[0]
        if len(pos):
            orden = pos[np.argsort(-a[pos], kind="stable")]
            acum = np.cumsum(a[orden]) / a[orden].sum()
            clase[orden, t] = np.where(acum <= .80, "A", np.where(acum <= .95, "B", "C"))
        clase[(edad[:, t] >= 0) & (a <= 0), t] = "C"
    return anual, clase


def fase_ciclo(datos):
    """(fase, r de tendencia) SKU × mes. Fase "" = SKU aún no lanzado."""
    y = np.nan_to_num(datos.y)
    edad, _ = edad_meses(datos)
    post = (edad >= 1).astype(float)                       # meses posteriores al lanzamiento
    yp = y * post
    s6, s12 = _suma_movil(y, 6), _suma_movil(y, N_DORMIDO)
    sb, cb = _rezago(_suma_movil(yp, 6), 12), _rezago(_suma_movil(post, 6), 12)     # mismos 6 meses, año anterior
    sa, ca = _rezago(np.cumsum(yp, axis=1), 6), _rezago(np.cumsum(post, axis=1), 6)  # alternativa: post-lanzamiento
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.where(cb > 0, sb / np.maximum(cb, 1), sa / np.maximum(ca, 1))
        r = np.where(base > 0, (s6 / 6) / base, np.inf)
    fase = np.full(y.shape, "", dtype=object)
    madura = edad >= 12
    fase[madura & (r >= UMBRAL_TEND)] = "crecimiento"
    fase[madura & (r < UMBRAL_TEND) & (r > 1 / UMBRAL_TEND)] = "madurez"
    fase[madura & ((r <= 1 / UMBRAL_TEND) | (s6 == 0))] = "declive"
    fase[madura & (s12 == 0)] = "dormido"
    fase[(edad >= 0) & (edad < 12)] = "introducción"
    return fase, np.where(madura & (s6 > 0), r, np.nan)


def niveles(abc, fase, base, ajustes=AJUSTE_FASE):
    """Matriz de niveles α: `base` (número o dict ABC → α) y, si se pasa `fase`, los `ajustes` en la escalera."""
    if isinstance(base, dict):
        lvl = np.vectorize(lambda c: base.get(c, .90), otypes=[float])(abc)
    else:
        lvl = np.full(abc.shape, float(base))
    if fase is None:
        return lvl
    i = np.searchsorted(ESCALERA, lvl)
    for f, d in ajustes.items():
        m = fase == f
        i[m] = 0 if d is None else np.clip(i[m] + d, 1, len(ESCALERA) - 1)
    return np.asarray(ESCALERA)[i]


class Segmentos:
    """Etiquetas ABC móvil y fase alineadas a las filas de un `inventario.Insumos`."""

    def __init__(self, datos):
        self.factor = factor_lanzamiento(datos)
        precio = matriz_precio(datos.skus)
        self.valor, self.abc = abc_movil(datos, precio, self.factor)
        self.valor_crudo, self.abc_crudo = abc_movil(datos, precio, self.factor, anualizar=False)
        self.fase, self.r = fase_ciclo(datos)
        self.edad, _ = edad_meses(datos)
        self.skus = datos.skus

    def para(self, ins):
        f = self.skus.get_indexer(ins.skus)
        return self.abc[f], self.fase[f]

    def tabla(self):
        n, T = self.abc.shape
        return pd.DataFrame({"id_articulo": np.repeat(self.skus.to_numpy(), T), "mes": np.tile(MESES, n),
                             "edad": self.edad.ravel(), "valor_anual": self.valor.ravel(),
                             "abc_movil": self.abc.ravel(), "abc_sin_anualizar": self.abc_crudo.ravel(),
                             "fase": self.fase.ravel(), "r_tendencia": self.r.ravel()}).query("abc_movil != ''")


def metricas_mascara(ins, sim, desde, hasta, mascara):
    """Como `inventario.metricas`, pero con una máscara SKU × mes (segmento que cambia en el tiempo)."""
    dem, ven, inv, ped = sim
    m = np.arange(iv.I(pd.Timestamp(desde)), iv.I(pd.Timestamp(hasta)) + 1)
    k = mascara[:, m]
    d, v, s = dem[:, m] * k, ven[:, m] * k, inv[:, m] * k
    p = ins.precio[:, None]
    con_dem = d > 0
    return {"fill_rate": v.sum() / d.sum(), "pct_meses_quiebre": ((v < d) & con_dem).sum() / con_dem.sum(),
            "stock_prom_mxn": (s * p).sum(axis=0).mean(), "cobertura_meses": (s * p).sum() / (d * p).sum(),
            "venta_perdida_mxn": ((d - v) * p).sum(axis=0).mean(), "sku_mes": int(k.sum())}


def politicas(abc, fase):
    pol = [(f"Uniforme {a:.0%}", a) for a in UNIFORMES]
    pol += [("ABC estático 95/90/80", iv.NIVEL_ABC),
            ("ABC móvil 95/90/80", niveles(abc, None, ABC_959080)),
            ("ABC móvil + fase", niveles(abc, fase, ABC_959080)),
            ("Uniforme 90% + fase", niveles(abc, fase, .90)),
            ("Uniforme 95% + fase", niveles(abc, fase, .95))]
    # descomposición: un solo ajuste de fase a la vez sobre el uniforme 90 %
    for f, d in AJUSTE_FASE.items():
        if d != 0:
            pol.append((f"Uniforme 90% · solo {f} {'sin SS' if d is None else f'{d:+d}'}",
                        niveles(abc, fase, .90, {f: d})))
    # ABC inverso: subir B (el segmento donde un punto de servicio cuesta menos stock) y dejar A en 90 %
    pol += [("ABC móvil 90/95/90", niveles(abc, None, {"A": .90, "B": .95, "C": .90})),
            ("ABC móvil 90/95/80", niveles(abc, None, {"A": .90, "B": .95, "C": .80}))]
    return pol


def main():
    t0 = time.time()
    datos = ft.Datos.cargar()
    seg = Segmentos(datos)
    seg.tabla().to_parquet(PROC / "segmentacion_sku_mes.parquet", index=False)
    print(f"FACTOR_LANZ = {seg.factor:.2f}")
    ins = iv.Insumos("ML_base", datos)
    abc, fase = seg.para(ins)
    # etiqueta del mes de demanda = la de la última decisión antes del mes (t − 1)
    abc_d, fase_d = _rezago_obj(abc), _rezago_obj(fase)
    filas, segm = [], []
    for nombre, alfa in politicas(abc, fase):
        sim = iv.simular(ins, "empirico", alfa)
        for periodo, (a, b) in iv.PERIODOS.items():
            filas.append({"politica": nombre, "periodo": periodo, **iv.metricas(ins, sim, a, b),
                          "stock_riesgo_mxn": iv.stock_en_riesgo(ins, sim) if periodo == "prueba" else np.nan})
            for dim, lab in (("abc_movil", abc_d), ("fase", fase_d)):
                for g in sorted(set(np.unique(lab)) - {""}):
                    segm.append({"politica": nombre, "periodo": periodo, "dimension": dim, "segmento": g,
                                 **metricas_mascara(ins, sim, a, b, lab == g)})
        print(f"  {nombre} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(filas)
    res.to_csv(SALIDA / "inv_segmentacion.csv", index=False)
    pd.DataFrame(segm).to_csv(SALIDA / "inv_segmentacion_seg.csv", index=False)
    print(res.round(3).to_string(index=False))


def _rezago_obj(a):
    out = np.full_like(a, "")
    out[:, 1:] = a[:, :-1]
    return out


if __name__ == "__main__":
    main()
