"""Pedido recomendado al cierre del último mes con datos (sep-2026), con la política elegida en LOG #25.

Ventana = oct-2026 … ene-2027 (R + L = 4). Cada mes se pronostica a h = 4 desde su origen (jun … sep-2026), igual que
en la simulación. Para los SKUs maduros se usa `TD_mix` (LOG #32): `ML_base` reconciliado con el nivel × índice
estacional de su familia (`jerarquico.py`). Para los jóvenes, `Media_post`. El SS es empírico, con todos los errores
de ventana de `TD_mix` observados hasta sep-2026.

    S = ventana + SS                      nivel objetivo (order-up-to)
    Q = max(0, S − on-hand − tránsito)    tránsito desconocido (§14 [SUPUESTO]): Q es una cota superior

On-hand: el catálogo trae el stock por almacén (todas las columnas ICM/AL + `extranjero`) al 21-may-2026, no a
sep-2026 (LOG #30). Sin recepciones ni órdenes abiertas, el on-hand al cierre de sep-2026 se acota por abajo con
max(0, stock 21-may − venta real 21-may…30-sep), así que Q es una cota superior.

Uso:  venv\\Scripts\\python.exe src\\pedido.py          (escribe reports/pedido_recomendado_2026-09.csv)
"""
import numpy as np
import pandas as pd

import json

import cold_start as cs
import features as ft
import inventario as iv
import jerarquico as jq
from backtest import HIPER, ml
from features import BASE, H, MESES, PROC, RAIZ
from metricas import clase_sb

NIVELES = (.90, .95)
ALMACENES = ["extranjero", "AL00001", "AL00002", "ICM01", "ICM02", "ICM03", "ICM04", "ICM06", "ICM07", "ICM08"]
FECHA_STOCK = pd.Timestamp("2026-05-21")   # fecha de la foto de stock del catálogo (confirmado por el usuario)


def main():
    datos = ft.Datos.cargar()
    meta = (PROC / "resultados" / "meta.json")
    hiper = HIPER[json.loads(meta.read_text(encoding="utf-8"))["hiper_elegido"]]
    k = json.loads((PROC / "resultados" / "meta_jer.json").read_text(encoding="utf-8"))["k_elegido"]
    modelo = f"TD_mix_{k}"
    tabla = pd.read_parquet(PROC / "features.parquet")
    T = len(MESES) - 1                                           # sep-2026
    origenes = range(T - H + 1, T + 1)                           # jun … sep-2026

    mad = []
    for o in origenes:
        te = tabla[tabla.origen == MESES[o]]
        f = ml(tabla, BASE, o, hiper)
        mad.append(pd.DataFrame({"id_articulo": te.id_articulo.to_numpy(), "familia": te.familia.to_numpy(),
                                 "origen": MESES[o], "mes_obj": te.mes_obj.to_numpy(), "ML_base": f.to_numpy(),
                                 "escala": te.escala.to_numpy(),
                                 "clase": clase_sb(te.adi.to_numpy(), te.cv2.to_numpy())}))
    mad = jq.construir(datos, pd.concat(mad, ignore_index=True))
    mad = [mad.rename(columns={modelo: "f"})[["id_articulo", "origen", "mes_obj", "f", "escala", "clase"]]]
    L, A = cs.lanzamientos(datos)
    jov, _ = cs.construir(datos, L, A, list(origenes))
    jov = jov[["id_articulo", "origen", "mes_obj", "Media_post", "escala"]].rename(columns={"Media_post": "f"})
    pron = pd.concat(mad + [jov.assign(clase="joven")], ignore_index=True)

    ancho = pron.pivot(index="id_articulo", columns="mes_obj", values="f")
    ancho.columns = [f"pron_{c:%Y-%m}" for c in ancho.columns]
    ultimo = pron[pron.origen == MESES[T]].set_index("id_articulo")[["escala", "clase", "f"]]
    # SKU que no estaba en el set en algún origen previo → proxy plano con su pronóstico del último origen (como en la simulación)
    ancho = ancho.apply(lambda c: c.fillna(ultimo.f.reindex(ancho.index))).fillna(0)
    out = ancho.join(ultimo[["escala", "clase"]])
    out["ventana"] = ancho.sum(axis=1)

    err = iv._errores_ventana(iv.Insumos(modelo, datos), T)
    vivo = out.escala.notna()
    for a in NIVELES:
        n = int(a * 100)
        out[f"ss_{n}"] = np.where(vivo, iv.ss_empirico(err, out.clase.to_numpy(), out.escala.to_numpy(), a), 0.0)
        out[f"S_{n}"] = out.ventana + out[f"ss_{n}"]

    cat = pd.read_csv(RAIZ / "data" / "cat_articulos_202610010754.csv", encoding="latin-1",
                      usecols=["id_articulo"] + ALMACENES).set_index("id_articulo")
    abc = pd.read_parquet(PROC / "clasificacion_sku.parquet").set_index("id_articulo").abc.astype(object)
    out["familia"] = pd.Series(datos.familia, index=datos.skus).reindex(out.index)
    out["abc"] = abc.reindex(out.index).fillna("sin_ABC")
    out["stock_21may"] = cat[ALMACENES].fillna(0).sum(axis=1).reindex(out.index).fillna(0)
    # venta real del 21-may al 30-sep (fracción de mayo por días) → on-hand mínimo al cierre de sep sin recepciones
    y = pd.DataFrame(np.nan_to_num(datos.y), index=datos.skus, columns=MESES).reindex(out.index).fillna(0)
    i = MESES.get_loc(FECHA_STOCK.to_period("M").to_timestamp())
    dias = FECHA_STOCK.days_in_month - FECHA_STOCK.day + 1
    out["venta_21may_sep"] = y.iloc[:, i] * dias / FECHA_STOCK.days_in_month + y.iloc[:, i + 1:T + 1].sum(axis=1)
    out["onhand_min_sep"] = np.maximum(out.stock_21may - out.venta_21may_sep, 0)
    for a in NIVELES:
        n = int(a * 100)
        out[f"Q_{n}_max"] = np.ceil(np.maximum(out[f"S_{n}"] - out.onhand_min_sep, 0))
    cols = ["familia", "abc", "clase"] + list(ancho.columns) + ["ventana"] + \
           [f"{p}_{int(a * 100)}" for a in NIVELES for p in ("ss", "S")] + \
           ["stock_21may", "venta_21may_sep", "onhand_min_sep"] + [f"Q_{int(a * 100)}_max" for a in NIVELES]
    out = out[cols].round(1).sort_values("ventana", ascending=False)
    dest = RAIZ / "reports" / f"pedido_recomendado_{MESES[T]:%Y-%m}.csv"
    out.to_csv(dest, encoding="utf-8-sig")
    print(f"{dest.name}: {len(out)} SKUs")
    print(out[["ventana", "S_90", "S_95", "stock_21may", "venta_21may_sep", "onhand_min_sep", "Q_90_max",
               "Q_95_max"]].sum().round(0).to_string())
    print(out.groupby("clase")[["ventana", "ss_95"]].sum().round(0).to_string())


if __name__ == "__main__":
    main()
