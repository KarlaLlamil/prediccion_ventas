"""XGBoost por serie (`notebooks/XGBoost.py`, LOG #33) para **todos** los SKUs del set evaluado, no solo el top 30.

Mismo modelo y features que el script del usuario (un `XGBRegressor` por SKU: mes + lags 3/4/5, 100 árboles,
profundidad 3) y mismo walk-forward h = 3: en cada origen T se entrena con la serie del SKU ≤ T y se pronostica T + 3.
Se calcula exactamente en los pares (SKU, origen) del backtest de validación y prueba, para compararlo contra los
demás modelos en las mismas filas. Control: en el top 30 reproduce `xgb_top30_sku.parquet`.

Uso:  venv\\Scripts\\python.exe src\\xgb_todos.py      (escribe data/processed/resultados/xgb_todos_*.parquet)
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest import ORIGENES, SALIDA
from features import H, MESES, PROC, RAIZ

sys.path.insert(0, str(RAIZ / "notebooks"))
import XGBoost as xg                                       # noqa: E402  (script del usuario)

LLAVES = ["id_articulo", "origen"]


def por_sku(serie, origenes):
    """Pronósticos XGBoost de un SKU para los orígenes pedidos (NaN si la historia no alcanza, como el original)."""
    prep = xg.preparar_datos_supervisados(serie)
    out = []
    for T in origenes:
        tr = prep[prep.ds <= T]
        te = prep[prep.ds == T + pd.DateOffset(months=H)]
        f = np.nan
        if len(tr) >= 5 and not te.empty:
            f = max(0, float(xg.nuevo_modelo().fit(tr[xg.FEATURES], tr.y).predict(te[xg.FEATURES])[0]))
        out.append(f)
    return out


def main():
    t0 = time.time()
    panel = pd.read_parquet(PROC / "panel.parquet", columns=["id_articulo", "mes", "unidades"])
    series = {k: g.rename(columns={"mes": "ds", "unidades": "y"})[["ds", "y"]].sort_values("ds")
              for k, g in panel.groupby("id_articulo")}
    for p in ORIGENES:
        pares = pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet", columns=LLAVES)
        grupos = list(pares.groupby("id_articulo").origen)
        res = Parallel(n_jobs=-1, batch_size=8)(delayed(por_sku)(series[s], list(o)) for s, o in grupos)
        out = pd.concat([pd.DataFrame({"id_articulo": s, "origen": list(o), "XGBoost": f})
                         for (s, o), f in zip(grupos, res)], ignore_index=True)
        out.to_parquet(SALIDA / f"xgb_todos_{p}.parquet", index=False)
        print(f"  {p}: {len(out)} pares, {out.XGBoost.isna().sum()} sin pronóstico ({time.time() - t0:.0f}s)", flush=True)

        top = pd.read_parquet(SALIDA / "xgb_top30_sku.parquet").query("periodo == @p")
        chk = top.merge(out, on=LLAVES, suffixes=("_top30", ""))
        assert len(chk) == len(top) and np.allclose(chk.XGBoost_top30, chk.XGBoost, rtol=1e-5), \
            f"{p}: no reproduce el XGBoost del top 30"
        print(f"  control top 30 ({p}): {len(chk)} pares idénticos", flush=True)


if __name__ == "__main__":
    main()
