"""Genera notebooks/06_td_lightgbm.ipynb a partir de las salidas de src/td_lgbm.py.

Uso (desde la raíz): venv\\Scripts\\python.exe src\\comparar_lgbm.py
                     venv\\Scripts\\python.exe src\\td_lgbm.py
                     venv\\Scripts\\python.exe notebooks\\_construir_td_lightgbm.py
Luego: venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace notebooks\\06_td_lightgbm.ipynb
"""
from pathlib import Path

import nbformat as nbf

CELDAS = []
md = lambda s: CELDAS.append(nbf.v4.new_markdown_cell(s.strip()))
py = lambda s: CELDAS.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 06 · `TD_estado` con LightGBM Poisson

Continuación de `05_lightgbm.ipynb`. Aquí se usa LightGBM con la **misma pérdida Poisson** que `ML_base` y se
reconstruye `TD_estado` (LOG #45) con ese LightGBM en lugar de `ML_base`. Luego se compara con `ML_base`,
`TD_estado` y el resto de los modelos del backtest.

**Qué cambia dentro de `TD_estado`.** `jerarquico.construir` usa `ML_base` en tres lugares, y en `TD_estado_LGBM`
cambian los tres:

| Pieza | `TD_estado` | `TD_estado_LGBM` |
|---|---|---|
| Familia **estable** | `ML_base` | LightGBM Poisson |
| Familia en **caída** → `TD_mix_k3` | (nivel 3 m × índice + Σ `ML_base`) / 2 × cuota `ML_base` | lo mismo con Σ y cuota de LightGBM |
| Familia en **subida** → `TD_mix_k12` | (nivel 12 m × índice + Σ `ML_base`) / 2 × cuota `ML_base` | lo mismo con Σ y cuota de LightGBM |

Los umbrales (0.85 / 1.15), el estado de cada familia, los niveles y los índices estacionales no cambian. Se reusa
la misma función, a la que se le pasa la tabla con `ML_base` sustituido. Como control, la misma función con
`ML_base` reproduce exactamente el `TD_estado` guardado. Código en `src/td_lgbm.py`; no se modifica nada de 01–05.

**LightGBM Poisson.** De las tres configuraciones Poisson de `05` (`L_A`, `L_bag`, `L_C`), la regla WAPE + sesgo en
validación frente a `L_A` elige **`L_C`**: lr 0.03, 600 árboles, 63 hojas, 30 obs. por hoja, columnas al 80 % y
L2 = 1. Una diferencia con `ML_base` que hay que tener presente: `ML_base` usa el *early stopping* automático de
scikit-learn, que con más de 10 mil filas aparta al azar un 10 % y se detiene en 42–63 árboles. LightGBM usa todos
sus árboles y el 100 % de las filas.
""")

py("""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from estilo import aplicar, COLORES, AZUL, GRIS, GRIS_CLARO, FMT_MILES
from metricas import resumen, adopta, por_origen, wape, sesgo

aplicar()
pd.set_option("display.width", 220, "display.max_columns", 40)
RES = RAIZ / "data" / "processed" / "resultados"
FIG = RAIZ / "reports" / "figuras"
LG = json.loads((RES / "meta_td_lgbm.json").read_text(encoding="utf-8"))["lgbm_poisson"]
PERIODOS = ["validacion", "prueba"]
d = {p: pd.read_parquet(RES / f"td_lgbm_{p}.parquet") for p in PERIODOS}
print(f"LightGBM Poisson elegido: {LG}")
print({p: len(x) for p, x in d.items()})
""")

md("""
## 1. Selección del LightGBM Poisson (solo validación)
""")
py("""
v = d["validacion"]
t = resumen(v, ["L_A", "L_bag", "L_C"])[["WAPE", "sesgo", "MASE"]]
t["gana_a_L_A"] = [np.nan if m == "L_A" else por_origen(v, [m, "L_A"]).pipe(lambda x: (x[m] < x.L_A).mean()) for m in t.index]
t["cumple_vs_L_A"] = [None if m == "L_A" else adopta(v, m, "L_A") for m in t.index]
print(t.round(3).to_string())
""")

md("""
## 2. Todos los modelos

Ordenados por WAPE. `gana` = orígenes en que el modelo tiene menor WAPE que la referencia. `cumple` = regla de
adopción (≥ 0.5 pp de WAPE, ≥ 2/3 de los orígenes y no más de 1 pp de |sesgo| extra). `L_tweedie` es el LightGBM
Tweedie de `05`.
""")
py("""
MOD = ["SNaive", "Naive", "MA12", "ETS", "Croston", "SBA", "TSB", "SARIMA_fam", "Ruta_clasica",
       "ML_base", LG, "L_tweedie", "TD_mix_k3", f"TD_mix_k3_LGBM", "TD_estado", "TD_estado_LGBM"]

def gana(x, m, ref):
    po = por_origen(x, [m, ref])
    return f"{(po[m] < po[ref]).sum()}/{len(po)}"

tablas = {}
for p in PERIODOS:
    x = d[p]
    r = resumen(x, MOD)[["WAPE", "sesgo", "MASE", "RMSSE", "WAPE_total"]]
    for ref in ("ML_base", "TD_estado"):
        r[f"gana vs {ref}"] = ["" if m == ref else gana(x, m, ref) for m in MOD]
        r[f"cumple vs {ref}"] = ["" if m == ref else adopta(x, m, ref) for m in MOD]
    tablas[p] = r
    print(f"===== {p}")
    print(r.sort_values("WAPE").round(3).to_string()); print()
""")

md("""
**Lectura.**

- **Validación:** `TD_estado_LGBM` tiene el menor WAPE de todos (0.438) y el menor sesgo de los modelos ML/TD
  (+9.5 %). Cumple la regla contra `ML_base`, pero **no contra `TD_estado`**: mejora el WAPE en 1.7 pp, pero solo en
  7 de 12 orígenes.
- **Prueba:** `TD_estado` sigue primero (0.482) y `TD_estado_LGBM` queda segundo, prácticamente empatado (0.483,
  8/13 orígenes, sesgo +2.0 vs +1.6 %). Contra `ML_base`, `TD_estado_LGBM` baja el WAPE en 0.7 pp, pero gana en solo
  8 de 13 orígenes (62 % < 2/3) y **no cumple la regla**. `TD_estado` sí la cumple (9/13).
- **LightGBM Poisson solo (`L_C`)** no le gana a `ML_base` en ninguno de los dos periodos (validación 7/12, prueba
  0.493 vs 0.490).
- **Los clásicos** (Ruta, SBA, MA12, TSB, Croston, SARIMA_fam, ETS, SNaive, Naive) quedan atrás en los dos periodos.
  La ruta clásica tiene buen sesgo en prueba (+1.7 %), pero WAPE 0.497 y peor MASE.
""")

md("### 2.1 Mano a mano de los cuatro candidatos")
py("""
C4 = ["ML_base", LG, "TD_estado", "TD_estado_LGBM"]
for p in PERIODOS:
    x = d[p]
    w = wape(x, C4)
    m = pd.DataFrame(index=C4, columns=C4, dtype=object)
    for a in C4:
        for b in C4:
            if a != b:
                po = por_origen(x, [a, b])
                m.loc[a, b] = f"{(w[a] - w[b]) * 100:+.1f} pp · {(po[a] < po[b]).sum()}/{len(po)}" + (" ✓" if adopta(x, a, b) else "")
    print(f"===== {p}  (fila vs columna: Δ WAPE · orígenes ganados · ✓ = cumple la regla)")
    print(m.fillna("—").to_string()); print()
""")

md("""
## 3. Por estado de la familia

Para ver qué pieza aporta: en las familias **estables** la diferencia es solo `ML_base` vs LightGBM. En **caída** y
**subida**, la diferencia está en la cuota y la Σ de `TD_mix`.
""")
py("""
for p in PERIODOS:
    x = d[p]
    g = x.groupby("estado_fam")[C4 + ["y"]]
    t = pd.DataFrame({f"WAPE {m}": g.apply(lambda z: wape(z, [m]).iloc[0]) for m in C4})
    for m in ("TD_estado", "TD_estado_LGBM"):
        t[f"sesgo {m}"] = g.apply(lambda z: sesgo(z, [m]).iloc[0])
    t["% unidades"] = x.groupby("estado_fam").y.sum() / x.y.sum()
    print(f"===== {p}"); print(t.round(3).to_string()); print()
""")

md("""
- **Validación:** casi toda la ventaja de `TD_estado_LGBM` está en las familias **estables** (55 % de las unidades).
  Ahí LightGBM baja el WAPE de 0.412 a 0.376 y el sesgo de +19 % a +11 %. En caída, la versión LightGBM recorta el
  sesgo (+10 % → +7 %), pero el WAPE queda casi igual (0.494 vs 0.488).
- **Prueba:** en estables los dos dan 0.457, un empate exacto. En caída y subida se mueven en sentidos opuestos y por
  poco. La ventaja de validación en las familias estables no se repite.
""")

md("## 4. Por origen")
py("""
fig, ax = plt.subplots(2, 1, figsize=(11, 6.6))
estilo_m = {"ML_base": dict(color=GRIS, ls="--"), LG: dict(color=COLORES[1], ls=":"),
            "TD_estado": dict(color=COLORES[2]), "TD_estado_LGBM": dict(color=AZUL)}
for a, p in zip(ax, PERIODOS):
    po = por_origen(d[p], C4)
    for m, kw in estilo_m.items():
        a.plot(po.index, po[m], marker="o", ms=3.5, label=m if m != LG else f"LightGBM Poisson ({LG})", **kw)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a.set_title(f"WAPE por origen · {p} (h = 3)")
ax[0].legend(fontsize=8, ncol=4)
ax[1].set_xlabel("Origen")
fig.tight_layout()
fig.savefig(FIG / "34_td_lgbm_wape_origen.png"); plt.show()
""")

py("""
fig, ax = plt.subplots(figsize=(11, 3.6))
for i, p in enumerate(PERIODOS):
    po = por_origen(d[p], ["TD_estado", "TD_estado_LGBM"])
    dif = (po.TD_estado_LGBM - po.TD_estado) * 100
    ax.bar(dif.index, dif.values, width=20, color=np.where(dif < 0, AZUL, COLORES[1]))
ax.axhline(0, color=GRIS, lw=1)
ax.axvline(pd.Timestamp("2025-04-15"), color=GRIS_CLARO, lw=1, ls="--")
ax.text(pd.Timestamp("2024-04-01"), ax.get_ylim()[1] * .9, "validación", color=GRIS, fontsize=8)
ax.text(pd.Timestamp("2025-06-01"), ax.get_ylim()[1] * .9, "prueba", color=GRIS, fontsize=8)
ax.set_ylabel("pp de WAPE")
ax.set_title("TD_estado_LGBM − TD_estado por origen (azul = LightGBM mejor)")
fig.savefig(FIG / "35_td_lgbm_diferencia.png"); plt.show()
""")

md("""
Las barras alternan de signo en los dos periodos: no hay un patrón sostenido a favor de ninguna versión. La
diferencia más grande es **ago-2025** (prueba): +4.3 pp en contra de LightGBM, el mismo origen en que `L_tweedie`
también falló en `05`.
""")

md("## 5. Total mensual en prueba")
py("""
x = d["prueba"]
tot = x.groupby("mes_obj")[["y", "ML_base", "TD_estado", "TD_estado_LGBM"]].sum()
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(tot.index, tot.y, color="#0b0b0b", lw=2.4, label="Real (set evaluado)")
ax.plot(tot.index, tot.ML_base, color=GRIS, ls="--", label="ML_base")
ax.plot(tot.index, tot.TD_estado, color=COLORES[2], label="TD_estado")
ax.plot(tot.index, tot.TD_estado_LGBM, color=AZUL, label="TD_estado_LGBM")
ax.yaxis.set_major_formatter(FMT_MILES)
ax.set_title("Unidades por mes objetivo · prueba (h = 3)")
ax.legend(fontsize=8, ncol=4)
fig.savefig(FIG / "36_td_lgbm_total_prueba.png"); plt.show()
e = tot[["ML_base", "TD_estado", "TD_estado_LGBM"]].sub(tot.y, axis=0).div(tot.y, axis=0)
print("Error del total (pronóstico − real) / real:")
print(e.round(3).T.to_string())
print("\\nWAPE del total mensual:", {m: round(abs(e[m] * tot.y).sum() / tot.y.sum(), 4) for m in e})
""")

md("""
## 6. Hallazgos y decisión

1. **LightGBM Poisson por sí solo no mejora a `ML_base`.** La mejor configuración Poisson (`L_C`) queda 2.1 pp
   abajo en validación, pero solo en 7/12 orígenes, y +0.3 pp arriba en prueba. Esto confirma lo de `05`: con la
   misma pérdida, LightGBM y HistGradientBoosting son equivalentes, aunque LightGBM use 600 árboles y `ML_base` se
   detenga en 42–63.
2. **`TD_estado_LGBM` es el mejor modelo de validación** (WAPE 0.438, sesgo +9.5 %), sobre todo por las familias
   estables. **En prueba empata con `TD_estado`** (0.483 vs 0.482, 8/13 orígenes) y no cumple la regla contra
   `ML_base` (8/13 < 2/3), mientras que `TD_estado` sí la cumple (9/13).
3. **La estructura pesa más que el regresor.** Pasar de `ML_base` a `TD_estado` mejora en los dos periodos, sea cual
   sea el regresor: −1.4 pp / −0.8 pp con HistGradientBoosting y −1.0 pp / −1.0 pp con LightGBM, comparando cada
   `TD_estado` con su propio ML (validación / prueba). Cambiar el regresor dentro de `TD_estado` mueve −1.7 pp en
   validación y +0.1 pp en prueba, sin una dirección estable.
4. **Decisión: `TD_estado` con `ML_base` se mantiene** como pronóstico de los SKUs maduros. Las dos versiones empatan
   en prueba y la vigente es la que cumple la regla en los dos periodos. Cambiar obligaría a regenerar inventario y
   pedido sin una mejora demostrada. `TD_estado_LGBM` queda como **alternativa equivalente** y candidato en el
   reentrenamiento mensual. Si en los próximos meses gana con margen, el cambio es directo: `td_lgbm.py` ya lo
   construye con la misma función.
""")

md("""
## 7. Variante Tweedie: `TD_estado_LGBM_tw`

Igual que `TD_estado_LGBM`, pero con el LightGBM **Tweedie** de `05` (`L_tweedie`, potencia 1.5, elegido en
validación entre las 4 configuraciones). Se cambian las mismas tres piezas: la rama estable y la cuota y Σ ML de
`TD_mix_k3` / `TD_mix_k12`. Recordatorio de `05`: Tweedie recorta el sobre-pronóstico y mejora a los SKUs chicos e
intermitentes.
""")
py("""
C6 = ["ML_base", LG, "L_tweedie", "TD_estado", "TD_estado_LGBM", "TD_estado_LGBM_tw"]
for p in PERIODOS:
    x = d[p]
    r = resumen(x, C6)[["WAPE", "sesgo", "MASE", "RMSSE", "WAPE_total"]]
    for ref in ("ML_base", "TD_estado"):
        r[f"gana vs {ref}"] = ["" if m == ref else gana(x, m, ref) for m in C6]
        r[f"cumple vs {ref}"] = ["" if m == ref else adopta(x, m, ref) for m in C6]
    print(f"===== {p}"); print(r.round(4).to_string()); print()
""")

md("### 7.1 Por estado de la familia y por origen")
py("""
C3 = ["TD_estado", "TD_estado_LGBM", "TD_estado_LGBM_tw"]
for p in PERIODOS:
    x = d[p]
    g = x.groupby("estado_fam")[C3 + ["y"]]
    t = pd.DataFrame({f"WAPE {m}": g.apply(lambda z: wape(z, [m]).iloc[0]) for m in C3})
    for m in C3:
        t[f"sesgo {m}"] = g.apply(lambda z: sesgo(z, [m]).iloc[0])
    t["% unidades"] = x.groupby("estado_fam").y.sum() / x.y.sum()
    print(f"===== {p}"); print(t.round(3).to_string()); print()
""")
py("""
fig, ax = plt.subplots(figsize=(11, 3.6))
for p in PERIODOS:
    po = por_origen(d[p], ["TD_estado", "TD_estado_LGBM_tw"])
    dif = (po.TD_estado_LGBM_tw - po.TD_estado) * 100
    ax.bar(dif.index, dif.values, width=20, color=np.where(dif < 0, AZUL, COLORES[1]))
ax.axhline(0, color=GRIS, lw=1)
ax.axvline(pd.Timestamp("2025-04-15"), color=GRIS_CLARO, lw=1, ls="--")
ax.text(pd.Timestamp("2024-04-01"), ax.get_ylim()[1] * .9, "validación", color=GRIS, fontsize=8)
ax.text(pd.Timestamp("2025-06-01"), ax.get_ylim()[1] * .9, "prueba", color=GRIS, fontsize=8)
ax.set_ylabel("pp de WAPE")
ax.set_title("TD_estado_LGBM_tw − TD_estado por origen (azul = Tweedie mejor)")
fig.savefig(FIG / "37_td_lgbm_tw_diferencia.png"); plt.show()
""")

md("""
### 7.2 Lectura

| | Validación | Prueba |
|---|---|---|
| WAPE `TD_estado` → `TD_estado_LGBM_tw` | 0.456 → **0.436** (9/12 orígenes) ✓ | 0.482 → 0.483 (9/13 orígenes) ✗ por WAPE |
| Sesgo | +15.2 % → +9.3 % | +1.6 % → +2.0 % |
| MASE | 0.771 → 0.703 | 0.942 → **0.924** (el mejor de todos los modelos) |
| Contra `ML_base` | cumple (8/12) | **cumple** (−0.7 pp, 10/13, sesgo +2.0 vs +3.5 %) |

1. **Es la mejor variante con LightGBM.** Supera a `TD_estado_LGBM` (Poisson) en los dos periodos, aunque por poco
   (0.436 vs 0.438 y 0.4831 vs 0.4832), y tiene el mejor MASE de todo el backtest en los dos periodos.
2. **Es el único modelo, aparte de `TD_estado`, que cumple la regla contra `ML_base` en validación y en prueba.**
   Además, en validación cumple la regla contra el propio `TD_estado`.
3. **En prueba empata con `TD_estado`:** gana en 9 de 13 orígenes, pero el WAPE total queda 0.1 pp arriba. Sus
   victorias por origen son chicas y sus derrotas más grandes (ago-2025 +3.2 pp, sep-2025 +1.4 pp). La regla exige
   0.5 pp de mejora, así que no la cumple.
4. **A favor del cambio:** su selección es más limpia. Tweedie y su potencia se eligieron solo con validación,
   mientras que la rama de subidas de `TD_estado` se definió viendo prueba (riesgo anotado en LOG #45). También es
   mejor en los SKUs chicos e intermitentes (MASE).
5. **En contra:** no mejora el WAPE ni el sesgo en prueba, agrega una dependencia (LightGBM) y obliga a regenerar
   inventario y pedido.

**Decisión pendiente del usuario.** Con la regla vigente, `TD_estado` se mantiene porque en prueba no hay mejora de
WAPE. `TD_estado_LGBM_tw` es la alternativa más sólida probada hasta ahora y el primer candidato para el
reentrenamiento mensual. Antes de cambiar, conviene comparar el inventario (stock a igual fill rate), como se hizo
con `TD_estado` en `03_inventario`.
""")

nb = nbf.v4.new_notebook()
nb.cells = CELDAS
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("06_td_lightgbm.ipynb"))
print("notebooks/06_td_lightgbm.ipynb escrito")
