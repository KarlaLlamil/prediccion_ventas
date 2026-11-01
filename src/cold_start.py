"""Cold start: SKUs jóvenes y pre-lanzamiento (LOG_DECISIONES #24; plan en reports/FEATURES.md §Cold start).

Edad a = mes − mes de lanzamiento (a = 0 es el mes de lanzamiento, que es llenado de canal ≈ 4× el nivel posterior).
Dos segmentos fuera del set maduro (≥ 13 meses de vida) del backtest principal:

    joven       0 ≤ a ≤ 11 en el origen T (al menos el mes de lanzamiento observado); objetivo en a + H ∈ [3, 14].
                Modelo global propio (HistGradientBoosting Poisson sobre y/escala) contra baselines simples.
    prelanz     lanzamiento en T+1 … T+H: sin venta propia en el origen; objetivo en a ∈ [0, H − 1] = [0, 2].
                Regla de análogos con rango p25–p75 / p10–p90 (supone que el lanzamiento se conoce por comercial).

(H = 3.) Análogos (causales): SKUs lanzados desde 2016 (los de 2015 están censurados por la izquierda) cuya curva de
A_MAX = 12 + H = 15 meses
ya terminó en el origen. Jerarquía familia+sublínea+banda de precio → familia+sublínea → familia → global, con ≥ 5
análogos por nivel. Atributos del catálogo: solo códigos (sublínea) y PrecioLista; nunca descripciones (LOG #2).

Escala de un SKU joven (contracción tipo bayesiana hacia el prior de análogos):
    escala = (n_post · media_post + K · nivel_análogos) / (n_post + K),  n_post = meses observados tras el lanzamiento.

Uso:  venv\\Scripts\\python.exe src\\cold_start.py          (escribe data/processed/resultados/cold_*.parquet)
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features as ft
from features import H, MESES, RAIZ, _por_familia, indice_estacional
from backtest import ORIGENES, PUENTE, SALIDA, _matriz, entrenamiento

warnings.filterwarnings("ignore")
EDAD_MAX = 12                          # meses de vida en el origen (a = 0…11)
A_MAX = EDAD_MAX + H                   # largo de la curva de análogos (a = 0…14 con H = 3)
NAC_MIN = 12                           # análogos lanzados desde ene-2016
MIN_ANALOGOS = 5
K = 3                                  # peso (en meses) del prior en la escala
NIVELES = [["familia", "sublinea", "banda"], ["familia", "sublinea"], ["familia"], []]
HIPER = dict(learning_rate=.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)
FEATS = ["edad", "n_post", "u_lanz_n", "media_post_n", "ultimo_n", "ma3_post_n", "frac_venta_post",
         "meses_sin_venta", "log_escala", "prior_nivel_n", "prior_obj_n", "prior_p25_n", "prior_p75_n", "prior_rel",
         "prior_superv", "prior_rampa", "log_n_analogos", "nivel_analogo", "log_precio", "ie_fam", "ie_total", "fam_momentum",
         "mes_obj_n", "familia"]
BASELINES = ["Media_post", "Escala", "Analogos", "Curva_analogos"]


def lanzamientos(datos):
    """Atributos de cada SKU para buscar análogos y su curva desde el lanzamiento (n_sku × A_MAX, NaN si no hay dato)."""
    cat = pd.read_csv(RAIZ / "data" / "cat_articulos_202610010754.csv", encoding="latin-1",
                      usecols=["id_articulo", "sublinea", "PrecioLista"]).set_index("id_articulo").reindex(datos.skus)
    precio = cat.PrecioLista.where(cat.PrecioLista > 0)
    banda = pd.qcut(np.log(precio), 3, labels=["p1", "p2", "p3"]).astype(object).fillna("p?")
    L = pd.DataFrame({"i_nac": datos.i_nac, "familia": datos.familia,
                      "sublinea": cat.sublinea.fillna(-1).astype(int).astype(str).to_numpy(),
                      "banda": banda.to_numpy(), "log_precio": np.log(precio).to_numpy()})
    n_mes = len(MESES)
    cols = L.i_nac.to_numpy()[:, None] + np.arange(A_MAX)[None, :]
    A = np.where(cols < n_mes, np.nan_to_num(datos.y)[np.arange(len(L))[:, None], np.minimum(cols, n_mes - 1)], np.nan)
    return L, A


_CACHE = {}


def _stats_analogos(L, A, o, lv):
    """Estadísticos por grupo de análogos (nivel lv de la jerarquía) con curvas terminadas en o. Memoizado."""
    if (o, lv) in _CACHE:
        return _CACHE[(o, lv)]
    elig = (L.i_nac >= NAC_MIN) & (L.i_nac + A_MAX - 1 <= o)
    E, Ae = L[elig], A[elig.to_numpy()]
    nivel = np.nanmean(Ae[:, 1:12], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(nivel[:, None] > 0, Ae / nivel[:, None], np.nan)
    acum = np.cumsum(Ae[:, 1:], axis=1) / np.arange(1, A_MAX)       # c{i}: media de las edades 1…i
    ed = pd.DataFrame({**{f"a{i}": Ae[:, i] for i in range(A_MAX)}, **{f"r{i}": rel[:, i] for i in range(A_MAX)},
                       **{f"v{i}": (Ae[:, i] > 0).astype(float) for i in range(A_MAX)},
                       **{f"c{i}": acum[:, i - 1] if i else np.full(len(Ae), np.nan) for i in range(A_MAX)},
                       "nivel": nivel}, index=E.index)
    claves = NIVELES[lv]
    g = ed.groupby([E[c] for c in claves]) if claves else ed.groupby(np.zeros(len(ed)))
    res = g.size(), {"med": g.median(), "q10": g.quantile(.1), "q25": g.quantile(.25), "q75": g.quantile(.75),
                     "q90": g.quantile(.9), "media": g.mean()}
    _CACHE[(o, lv)] = res
    return res


def analogos(L, A, o, filas, a_obj, a_obs=None):
    """Estadísticos de análogos en la edad objetivo para las filas (posiciones de SKU) dadas, con datos ≤ o.
    a_obs (jóvenes): última edad observada; `rampa` = media de análogos en a_obj / su media en las edades 1…a_obs."""
    F = L.loc[filas].reset_index(drop=True)
    out = {c: np.full(len(F), np.nan) for c in ["obj", "p10", "p25", "p75", "p90", "rel", "superv", "nivel", "n",
                                                "rampa"]}
    out["nivel_analogo"] = np.full(len(F), np.nan)
    pend = np.ones(len(F), bool)
    for lv, claves in enumerate(NIVELES):
        if not pend.any():
            break
        n, stats = _stats_analogos(L, A, o, lv)
        idx = pd.MultiIndex.from_frame(F[claves]) if len(claves) > 1 else (
            pd.Index(F[claves[0]]) if claves else pd.Index(np.zeros(len(F))))
        pos = n.index.get_indexer(idx)
        ok = pend & (pos >= 0)
        ok[ok] = n.to_numpy()[pos[ok]] >= MIN_ANALOGOS
        if not ok.any():
            continue
        def toma(tab, pref):
            m = tab[[f"{pref}{i}" for i in range(A_MAX)]].to_numpy()[pos[ok]]
            return m[np.arange(ok.sum()), a_obj[ok]]
        # punto = media (conserva el total; la mediana de curvas tan asimétricas sub-pronostica ~70 %);
        # los cuantiles solo dan el rango
        out["obj"][ok] = toma(stats["media"], "a")
        for q, c in [("q10", "p10"), ("q25", "p25"), ("q75", "p75"), ("q90", "p90")]:
            out[c][ok] = toma(stats[q], "a")
        nivel_m = stats["media"].nivel.to_numpy()[pos[ok]]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["rel"][ok] = np.where(nivel_m > 0, toma(stats["media"], "a") / nivel_m, np.nan)   # razón de sumas
        out["superv"][ok] = toma(stats["media"], "v")
        out["nivel"][ok] = nivel_m
        if a_obs is not None:
            m = stats["media"][[f"c{i}" for i in range(A_MAX)]].to_numpy()[pos[ok]]
            base = m[np.arange(ok.sum()), a_obs[ok]]
            with np.errstate(divide="ignore", invalid="ignore"):
                out["rampa"][ok] = np.where(base > 0, toma(stats["media"], "a") / base, np.nan)
        out["n"][ok] = n.to_numpy()[pos[ok]]
        out["nivel_analogo"][ok] = lv
        pend &= ~ok
    return out


def construir(datos, L, A, origenes):
    """Tablas (jóvenes, pre-lanzamiento) para los orígenes dados. Todo feature usa meses ≤ origen."""
    y0 = np.nan_to_num(datos.y)
    S = np.cumsum(y0, axis=1)
    NZ = np.cumsum(y0 > 0, axis=1)
    idx = np.where(y0 > 0, np.arange(len(MESES))[None, :], -1)
    ultima = np.maximum.accumulate(idx, axis=1)
    fams = _por_familia(datos, datos.y[:, :1])[0]
    pos_fam = pd.Index(fams).get_indexer(datos.familia)
    i_nac, n_mes = datos.i_nac, len(MESES)

    jov, pre = [], []
    for o in origenes:
        obj = o + H
        mes_obj = MESES[o] + pd.DateOffset(months=H)
        y_obj = lambda k: y0[k, obj] if obj < n_mes else np.full(len(k), np.nan)
        base = {"origen": MESES[o], "mes_obj": mes_obj, "obj_bajo": bool(datos.bajo[obj]) if obj < n_mes else False}
        # agregados de familia con la serie de modelado vista desde el origen (como features.construir)
        _, fc = _por_familia(datos, datos.serie(o)[:, :o + 1])
        fam_ma3, fam_ma12 = fc[:, -3:].mean(axis=1), fc[:, -12:].mean(axis=1)
        fc_df = pd.DataFrame(fc.T, index=MESES[:o + 1], columns=fams)
        tot_df = fc_df.sum(axis=1).to_frame("total")

        k = np.flatnonzero((o - i_nac >= 0) & (o - i_nac < EDAD_MAX))
        if len(k):
            a_o = o - i_nac[k]
            an = analogos(L, A, o, k, a_o + H, a_o)
            n_post = a_o
            u_lanz = y0[k, i_nac[k]]
            suma_post = S[k, o] - S[k, i_nac[k]]
            with np.errstate(divide="ignore", invalid="ignore"):
                media_post = np.where(n_post > 0, suma_post / np.maximum(n_post, 1), np.nan)
                n3 = np.minimum(n_post, 3)
                ma3 = np.where(n3 > 0, (S[k, o] - S[k, o - n3]) / np.maximum(n3, 1), np.nan)
            prior_nivel = np.nan_to_num(an["nivel"], nan=0.0)
            escala = np.maximum((n_post * np.nan_to_num(media_post) + K * prior_nivel) / (n_post + K), .05)
            f = pos_fam[k]
            ie_f = indice_estacional(fc_df)
            ie_tot = indice_estacional(tot_df)["total"].get(mes_obj.month, np.nan)
            ie_fam = ie_f.loc[mes_obj.month].to_numpy()[f] if mes_obj.month in ie_f.index else np.full(len(k), np.nan)
            r = {"id_articulo": datos.skus[k], "familia": datos.familia[k], "perfil": datos.perfil[k], **base,
                 "y": y_obj(k), "escala": escala, "edad": a_o + 1, "n_post": n_post,
                 "u_lanz_n": u_lanz / escala, "media_post_n": media_post / escala,
                 "ultimo_n": np.where(n_post > 0, y0[k, o], np.nan) / escala, "ma3_post_n": ma3 / escala,
                 "frac_venta_post": np.where(n_post > 0, (NZ[k, o] - NZ[k, i_nac[k]]) / np.maximum(n_post, 1), np.nan),
                 "meses_sin_venta": o - ultima[k, o], "log_escala": np.log(escala),
                 "prior_nivel_n": an["nivel"] / escala, "prior_obj_n": an["obj"] / escala,
                 "prior_p25_n": an["p25"] / escala, "prior_p75_n": an["p75"] / escala, "prior_rel": an["rel"],
                 "prior_superv": an["superv"], "prior_rampa": an["rampa"], "log_n_analogos": np.log(an["n"]), "nivel_analogo": an["nivel_analogo"],
                 "log_precio": L.log_precio.to_numpy()[k],
                 "ie_fam": np.where(np.isnan(ie_fam), ie_tot, ie_fam), "ie_total": ie_tot,
                 "fam_momentum": fam_ma3[f] / np.maximum(fam_ma12[f], 1e-3), "mes_obj_n": mes_obj.month,
                 # baselines
                 "Media_post": np.where(n_post > 0, media_post, prior_nivel), "Escala": escala,
                 "Analogos": escala * np.nan_to_num(an["rel"], nan=1.0),
                 "Curva_analogos": np.where(n_post > 0, np.nan_to_num(media_post) * np.nan_to_num(an["rampa"], nan=1.0),
                                            np.nan_to_num(an["obj"]))}
            jov.append(pd.DataFrame(r))

        k = np.flatnonzero((i_nac > o) & (i_nac <= obj))
        if len(k):
            an = analogos(L, A, o, k, obj - i_nac[k])
            pre.append(pd.DataFrame({"id_articulo": datos.skus[k], "familia": datos.familia[k],
                                     "perfil": datos.perfil[k], **base, "y": y_obj(k), "edad_obj": obj - i_nac[k],
                                     "Analogos_med": an["obj"], "p10": an["p10"], "p25": an["p25"], "p75": an["p75"],
                                     "p90": an["p90"], "nivel_analogo": an["nivel_analogo"]}))
    return pd.concat(jov, ignore_index=True), pd.concat(pre, ignore_index=True)


def ml(tabla, o):
    """Igual que backtest.ml: entrena con objetivos conocidos en o (sin meses bajos del total) y pronostica el origen o.
    El objetivo es crudo: un SKU joven todavía no tiene clase SB estable para decidir si se limpia."""
    T = MESES[o]
    tr = entrenamiento(tabla, T)
    te = tabla[tabla.origen == T]
    cats = sorted(tr.familia.unique())
    m = HistGradientBoostingRegressor(loss="poisson", categorical_features="from_dtype", random_state=0, **HIPER)
    m.fit(_matriz(tr, FEATS, cats), tr.y / tr.escala)
    return pd.Series(m.predict(_matriz(te, FEATS, cats)) * te.escala.to_numpy(), index=te.index)


def resumen_joven(df, modelos):
    """Sin 12 meses de historia no hay escala MASE: WAPE, sesgo, error en log y vista total (LOG #24)."""
    filas = {}
    for m in modelos:
        e = df[m] - df.y
        tot = df.groupby("mes_obj")[[m, "y"]].sum()
        filas[m] = {"WAPE": e.abs().sum() / df.y.sum(), "sesgo": e.sum() / df.y.sum(),
                    "MAE_log": (np.log1p(df[m]) - np.log1p(df.y)).abs().mean(),
                    "WAPE_total": (tot[m] - tot.y).abs().sum() / tot.y.sum(),
                    "sesgo_total": (tot[m] - tot.y).sum() / tot.y.sum()}
    return pd.DataFrame(filas).T


def resumen_prelanz(df):
    e = df.Analogos_med - df.y
    return pd.Series({"filas": len(df), "SKUs": df.id_articulo.nunique(), "unidades": df.y.sum(),
                      "WAPE": e.abs().sum() / df.y.sum(), "sesgo": e.sum() / df.y.sum(),
                      "cobertura_p25_p75": df.y.between(df.p25, df.p75).mean(),
                      "cobertura_p10_p90": df.y.between(df.p10, df.p90).mean()})


def main():
    t0 = time.time()
    datos = ft.Datos.cargar()
    L, A = lanzamientos(datos)
    primer = NAC_MIN + A_MAX - 1                       # primer origen con análogos completos
    jov, pre = construir(datos, L, A, list(range(primer, len(MESES))))
    print(f"jóvenes {jov.shape}, pre-lanzamiento {pre.shape}  ({time.time() - t0:.0f}s)", flush=True)

    meta = {}
    for periodo, origenes in {**ORIGENES, "puente": PUENTE}.items():     # puente: solo continuidad para inventario
        d = jov[jov.origen.isin(MESES[list(origenes)])].copy()
        d["ML_joven"] = pd.concat([ml(jov, o) for o in origenes])
        d.to_parquet(SALIDA / f"cold_jovenes_{periodo}.parquet", index=False)
        p = pre[pre.origen.isin(MESES[list(origenes)])]
        p.to_parquet(SALIDA / f"cold_prelanz_{periodo}.parquet", index=False)
        meta[periodo] = {"jovenes": resumen_joven(d, BASELINES + ["ML_joven"]).round(4).to_dict("index"),
                         "prelanz": resumen_prelanz(p).round(4).to_dict()}
    meta["segundos"] = round(time.time() - t0)
    (SALIDA / "meta_cold.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float),
                                           encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
