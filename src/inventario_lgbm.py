"""Inventario simulado con `TD_estado_LGBM_tw` contra `TD_estado` (`notebooks/07_inventario_lgbm.ipynb`).

Misma política y simulación de `inventario.py` (order-up-to R = 1, L = 2, SS empírico, venta perdida, calentamiento,
`Media_post` para jóvenes); solo cambia el pronóstico de los SKUs maduros. La simulación necesita los tres periodos
del backtest (validación, puente y prueba), así que aquí se completa el **puente** (orígenes abr–may 2025), que
`comparar_lgbm.py` / `td_lgbm.py` no calculan: LightGBM Tweedie con datos ≤ origen y `jerarquico.construir` con
`ML_base` sustituido, igual que en `td_lgbm.py`.

Modelos: `TD_estado` (vigente), `TD_estado_LGBM_tw`, y como contexto `ML_base` y `Ruta_clasica`.
Políticas: sin SS y SS empírico uniforme al 70 / 80 / 90 / 95 / 98 % (curva para comparar a igual fill rate).

Uso:  venv\\Scripts\\python.exe src\\inventario_lgbm.py     (escribe data/processed/resultados/inv_lgbm.csv)
"""
import time

import numpy as np
import pandas as pd

import features as ft
import inventario as inv
from backtest import PUENTE, SALIDA
from comparar_lgbm import HIPER, lgbm
from features import BASE, MESES
from jerarquico import ELEGIDO, construir

MODELO = "TD_estado_LGBM_tw"
MODELOS = ["TD_estado", MODELO, "ML_base", "Ruta_clasica"]
NIVELES = [.70, .80, .90, .95, .98]
LLAVES = ["id_articulo", "origen"]


def puente(datos):
    """`L_tweedie` y `TD_estado_LGBM_tw` para los orígenes del puente."""
    tabla = pd.read_parquet(ft.PROC / "features.parquet")
    bt = pd.read_parquet(SALIDA / "pronosticos_puente.parquet")
    p = pd.concat([lgbm(tabla, BASE, o, HIPER["L_tweedie"]).rename("L_tweedie").rename_axis("id_articulo")
                   .reset_index().assign(origen=MESES[o]) for o in PUENTE])
    bt = bt.merge(p, on=LLAVES, how="left", validate="1:1")
    assert bt.L_tweedie.notna().all(), "puente: filas sin LightGBM"
    td = construir(datos, bt.assign(ML_base=bt.L_tweedie))
    return td[LLAVES + [ELEGIDO]].rename(columns={ELEGIDO: MODELO})


def main():
    t0 = time.time()
    datos = ft.Datos.cargar()
    extra = pd.concat([pd.read_parquet(SALIDA / f"td_lgbm_{p}.parquet")[LLAVES + [MODELO]]
                       for p in ("validacion", "prueba")] + [puente(datos)], ignore_index=True)
    extra.to_parquet(SALIDA / "td_lgbm_tw_inventario.parquet", index=False)
    print(f"pronósticos {MODELO}: {len(extra)} filas ({time.time() - t0:.0f}s)", flush=True)

    filas = []
    politicas = [("sin_ss", 0)] + [("empirico", a) for a in NIVELES]
    for modelo in MODELOS:
        ins = inv.Insumos(modelo, datos, extra if modelo == MODELO else None)
        for metodo, alfa in politicas:
            sim = inv.simular(ins, metodo, alfa)
            for periodo, (a, b) in inv.PERIODOS.items():
                filas.append({"modelo": modelo, "metodo": metodo, "nivel": alfa or None, "periodo": periodo,
                              **inv.metricas(ins, sim, a, b),
                              "stock_riesgo_mxn": inv.stock_en_riesgo(ins, sim) if periodo == "prueba" else np.nan})
        print(f"  {modelo} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(filas)
    res.to_csv(SALIDA / "inv_lgbm.csv", index=False)
    print(res[res.metodo == "empirico"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
