"""LightGBM contra `ML_base` en el mismo walk-forward h = 3 (`notebooks/05_lightgbm.ipynb`).

Misma tabla de features (`features.parquet`), mismas columnas `BASE`, mismo objetivo (`y_entreno / escala`, sin los
meses bajos del total) y mismos orígenes que `backtest.py`; solo cambia el regresor. Así la diferencia se le puede
atribuir al algoritmo y no a los datos.

1) Hiperparámetros: solo en validación, regla WAPE + sesgo (`metricas.elegir`) frente a `L_A`, la traducción directa
   de los hiperparámetros A de `ML_base` (300 árboles, 31 hojas, 50 obs. por hoja, lr 0.05, Poisson).
2) La configuración elegida se compara con `ML_base` (columna ya guardada por `backtest.py`) con la misma regla, en
   validación y prueba. Las demás configuraciones también se corren en prueba, solo como reporte.

Uso:  venv\\Scripts\\python.exe src\\comparar_lgbm.py      (escribe data/processed/resultados/lgbm_*.parquet)
"""
import json
import time
import warnings

import lightgbm as lgb
import pandas as pd

import features as ft
from backtest import ORIGENES, SALIDA, _matriz, entrenamiento
from features import BASE, MESES
from metricas import elegir, resumen

warnings.filterwarnings("ignore")

COMUN = dict(random_state=0, n_jobs=-1, verbose=-1, deterministic=True, force_col_wise=True)
HIPER = {"L_A": dict(objective="poisson", learning_rate=.05, n_estimators=300, num_leaves=31, min_child_samples=50),
         # lo que LightGBM agrega sobre HistGradientBoosting: muestreo de filas y columnas + L2
         "L_bag": dict(objective="poisson", learning_rate=.05, n_estimators=300, num_leaves=31, min_child_samples=50,
                       subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_lambda=1.0),
         "L_C": dict(objective="poisson", learning_rate=.03, n_estimators=600, num_leaves=63, min_child_samples=30,
                     colsample_bytree=.8, reg_lambda=1.0),
         # Tweedie: masa en cero + cola positiva, pensado para demanda intermitente
         "L_tweedie": dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=.05, n_estimators=300,
                           num_leaves=31, min_child_samples=50)}


def lgbm(tabla, cols, o, hiper, importancia=False):
    """Igual que `backtest.ml`, con LGBMRegressor. Pronóstico de las filas del origen o (en unidades)."""
    T = MESES[o]
    tr = entrenamiento(tabla, T)
    te = tabla[tabla.origen == T]
    cats = sorted(tr.familia.unique())
    m = lgb.LGBMRegressor(**COMUN, **hiper)
    m.fit(_matriz(tr, cols, cats), tr.y_entreno / tr.escala)
    p = pd.Series(m.predict(_matriz(te, cols, cats)) * te.escala.to_numpy(), index=te.id_articulo.to_numpy())
    if importancia:
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
        return p, imp / imp.sum()
    return p


def correr(tabla, origenes, hiper):
    partes, seg = [], 0.0
    for o in origenes:
        t = time.time()
        p = lgbm(tabla, BASE, o, hiper)
        seg += time.time() - t
        partes.append(pd.DataFrame({"id_articulo": p.index, "origen": MESES[o], "f": p.to_numpy()}))
    return pd.concat(partes, ignore_index=True), seg / len(origenes)


def main():
    t0 = time.time()
    tabla = pd.read_parquet(ft.PROC / "features.parquet")
    res = {p: pd.read_parquet(SALIDA / f"pronosticos_{p}.parquet") for p in ORIGENES}
    llaves = ["id_articulo", "origen"]

    # 1) hiperparámetros: solo validación
    val = res["validacion"]
    hiper_res, seg_ajuste = {}, {}
    for nombre, h in HIPER.items():
        p, seg_ajuste[nombre] = correr(tabla, ORIGENES["validacion"], h)
        val = val.merge(p.rename(columns={"f": nombre}), on=llaves, how="left")
        hiper_res[nombre] = resumen(val, [nombre]).iloc[0].to_dict()
        print(f"  {nombre}: WAPE {hiper_res[nombre]['WAPE']:.3f}  sesgo {hiper_res[nombre]['sesgo']:+.3f}  "
              f"({seg_ajuste[nombre]:.1f} s por origen)", flush=True)
    assert val[list(HIPER)].notna().all().all(), "filas de validación sin pronóstico LightGBM"
    mejor = elegir(val, list(HIPER), "L_A")
    print(f"  elegido: {mejor}", flush=True)

    # 2) prueba: todas las configuraciones, solo para reporte (la elección ya quedó fija con validación). L_A en
    #    prueba aísla el efecto del algoritmo; L_A → elegido, el de los hiperparámetros/objetivo.
    prueba = res["prueba"]
    for nombre, h in HIPER.items():
        p, _ = correr(tabla, ORIGENES["prueba"], h)
        prueba = prueba.merge(p.rename(columns={"f": nombre}), on=llaves, how="left")
    assert prueba[list(HIPER)].notna().all().all(), "filas de prueba sin pronóstico LightGBM"

    # importancia (gain) del modelo elegido en el último origen de prueba, junto a la de cada config. en validación
    _, imp = lgbm(tabla, BASE, ORIGENES["prueba"][-1], HIPER[mejor], importancia=True)

    cols = llaves + ["mes_obj", "y", "familia", "clase", "escala_mae", "escala_mse", "SNaive", "ML_base"]
    for periodo, d in (("validacion", val), ("prueba", prueba)):
        d[cols + list(HIPER)].to_parquet(SALIDA / f"lgbm_{periodo}.parquet", index=False)
    imp.sort_values(ascending=False).rename("gain").to_frame().to_parquet(SALIDA / "lgbm_importancia.parquet")
    meta = {"hiper": {k: {**COMUN, **v} for k, v in HIPER.items()}, "resultado_validacion": hiper_res,
            "elegido": mejor, "segundos_por_origen": seg_ajuste, "lightgbm": lgb.__version__,
            "segundos": round(time.time() - t0)}
    (SALIDA / "meta_lgbm.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("elegido", "segundos")}, indent=2))


if __name__ == "__main__":
    main()
