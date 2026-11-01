"""`TD_estado` con LightGBM Poisson en lugar de `ML_base` (`notebooks/06_td_lightgbm.ipynb`).

`jerarquico.construir` usa la columna `ML_base` en tres lugares: la rama estable de `TD_estado`, la cuota de cada SKU
dentro de su familia (`TD_mix`) y la mitad del volumen de la familia (`TD_mix` = (nivel × índice + Σ ML) / 2 × cuota).
Aquí se le pasa la tabla del backtest con `ML_base` reemplazado por el LightGBM Poisson, así que las tres piezas
cambian. La regla (umbrales 0.85 / 1.15; caída → k3, subida → k12) y los niveles por familia quedan iguales.

Dos versiones, ambas de `comparar_lgbm.py` (hay que correrlo antes):
    `TD_estado_LGBM`     LightGBM Poisson: de `L_A`, `L_bag`, `L_C`, la que elige la regla WAPE + sesgo en validación
                         frente a `L_A`
    `TD_estado_LGBM_tw`  LightGBM Tweedie (`L_tweedie`)

Uso:  venv\\Scripts\\python.exe src\\td_lgbm.py      (escribe data/processed/resultados/td_lgbm_*.parquet)
"""
import json

import numpy as np
import pandas as pd

import features as ft
from backtest import ORIGENES, SALIDA
from jerarquico import ELEGIDO, construir
from metricas import elegir

POISSON = ["L_A", "L_bag", "L_C"]
LLAVES = ["id_articulo", "origen"]
VERSIONES = {"LGBM": None,                  # None → el Poisson elegido en validación
             "LGBM_tw": "L_tweedie"}        # Tweedie 1.5 (elegido en validación entre las 4 config. de `05`)


def main():
    datos = ft.Datos.cargar()
    lg = {p: pd.read_parquet(SALIDA / f"lgbm_{p}.parquet") for p in ORIGENES}
    mejor = elegir(lg["validacion"], POISSON, "L_A")
    lg = {p: d[LLAVES + POISSON + ["L_tweedie"]] for p, d in lg.items()}
    print(f"LightGBM Poisson elegido en validación: {mejor}", flush=True)

    meta = {"lgbm_poisson": mejor}
    for p in ORIGENES:
        bt = pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet").merge(lg[p], on=LLAVES, how="left", validate="1:1")
        assert bt[POISSON].notna().all().all(), f"{p}: filas sin LightGBM"
        jer = pd.read_parquet(SALIDA / f"pronosticos_{p}_jer.parquet")

        # control: construir con ML_base debe reproducir el TD_estado guardado
        ref = construir(datos, bt)
        assert np.allclose(ref.set_index(LLAVES).TD_estado.reindex(jer.set_index(LLAVES).index), jer.TD_estado), \
            f"{p}: TD_estado no se reproduce"

        out = bt.merge(jer[LLAVES + ["estado_fam", "TD_mix_k3", "TD_mix_k12", ELEGIDO]], on=LLAVES, validate="1:1")
        for sufijo, modelo in VERSIONES.items():
            lgb = construir(datos, bt.assign(ML_base=bt[modelo if modelo else mejor]))
            nuevos = lgb[LLAVES + ["TD_mix_k3", "TD_mix_k12", ELEGIDO]].rename(
                columns={c: f"{c}_{sufijo}" for c in ["TD_mix_k3", "TD_mix_k12", ELEGIDO]})
            out = out.merge(nuevos, on=LLAVES, validate="1:1")
        out.to_parquet(SALIDA / f"td_lgbm_{p}.parquet", index=False)
        meta[p] = len(out)
        print(f"  {p}: {len(out)} filas", flush=True)
    (SALIDA / "meta_td_lgbm.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
