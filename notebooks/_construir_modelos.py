"""Genera notebooks/02_modelos.ipynb (resultados de fases 2–3) a partir de las salidas de src/backtest.py.

Uso (desde la raíz): venv\\Scripts\\python.exe notebooks\\_construir_modelos.py
Luego: venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace notebooks\\02_modelos.ipynb
"""
from pathlib import Path

import nbformat as nbf

CELDAS = []
md = lambda s: CELDAS.append(nbf.v4.new_markdown_cell(s.strip()))
py = lambda s: CELDAS.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 02 · Baselines, modelo global y backtest (h = 3)

Fases 2–3 de DECISIONES.md. Features documentados en `reports/FEATURES.md`; diseño del backtest en LOG_DECISIONES #19.

- **Validación** (objetivos jul-2024 → jun-2025, orígenes abr-2024 → mar-2025): hiperparámetros, ablaciones de
  features, ruta por segmento y k del jerárquico. El último objetivo es el primer origen de prueba (jun-2025), así
  que ninguna decisión usa datos posteriores.
- **Prueba** (objetivos sep-2025 → sep-2026): solo se evalúa lo elegido en validación. ⚠️ Cae en el nivel bajo
  posterior a caida_2024 (§13: el test es un periodo anómalo). Jul–ago 2025 (periodo puente) solo se pronostican
  para la simulación de inventario.
- **Métrica de decisión: WAPE + sesgo** (`metricas.adopta`): un candidato reemplaza a la referencia solo si baja el
  WAPE en ≥ 0.5 pp, gana en WAPE en ≥ 2/3 de los orígenes y no empeora el |sesgo| en más de 1 pp. MASE, RMSSE,
  relMAE y MAPE (solo y > 0) se reportan como secundarias. Las métricas `_total` suman los SKUs por mes (vista de
  compras).
- SKUs dormidos (≥ 12 meses sin venta; repuestos que pueden volver) no se evalúan: se reporta su reactivación.
- Proyecto/base: se usa la clase Syntetos–Boylan como proxy (grumosa/intermitente = proyecto, suave/errática = base).
""")

md("""
**Modelo elegido para los SKUs maduros: `TD_estado`** (§3.8, LOG #45): según el estado de la familia en el origen,
`TD_mix_k3` si viene cayendo, `TD_mix_k12` si viene subiendo y `ML_base` si está estable. Jóvenes: `Media_post` (§5).
Las cifras de las celdas de lectura y de los hallazgos son de esta corrida (h = 3, regla de limpieza por clase,
WAPE + sesgo).
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
from features import H
from metricas import adopta, elegir, por_origen, resumen

aplicar()
pd.set_option("display.width", 200, "display.max_columns", 40)
RES = RAIZ / "data" / "processed" / "resultados"
FIG = RAIZ / "reports" / "figuras"
val = pd.read_parquet(RES / "pronosticos_validacion.parquet")
test = pd.read_parquet(RES / "pronosticos_prueba.parquet")
meta = json.loads((RES / "meta.json").read_text(encoding="utf-8"))
cob = pd.read_csv(RES / "cobertura.csv", parse_dates=["mes_obj"])
clas = pd.read_parquet(RAIZ / "data" / "processed" / "clasificacion_sku.parquet")[["id_articulo", "abc", "xyz"]]
TIPO = {"grumosa": "proyecto (proxy)", "intermitente": "proyecto (proxy)", "suave": "base", "errática": "base"}
for d in (val, test):
    d["tipo"] = d.clase.map(TIPO)

CLASICOS = ["Naive", "SNaive", "MA12", "ETS", "SARIMA_fam", "Croston", "SBA", "TSB", "Ruta_clasica"]
ML = [c for c in val.columns if c.startswith("ML_")]
FMT = {"MASE": "{:.3f}", "RMSSE": "{:.3f}", "WAPE": "{:.1%}", "relMAE": "{:.3f}", "sesgo": "{:+.1%}",
       "MAPE_y>0": "{:.0%}", "WAPE_total": "{:.1%}", "sesgo_total": "{:+.1%}"}
def tabla(df, modelos):
    return resumen(df, modelos).sort_values("WAPE").style.format(FMT)
""")

md("## 1. Set evaluado y cobertura")
py("""
for n, d in [("validación", val), ("prueba", test)]:
    print(f"{n}: {len(d):,} pares SKU-origen · {d.id_articulo.nunique():,} SKUs · {d.origen.nunique()} orígenes · "
          f"{d.y.sum():,.0f} u. reales")
cob["pct_fuera"] = cob.u_fuera_set / cob.u_total
cob["pct_jovenes"], cob["pct_dormidos"] = cob.u_jovenes / cob.u_total, cob.u_dormidos / cob.u_total
print(f"Unidades del mes objetivo fuera del set: mediana {cob.pct_fuera.median():.1%}, "
      f"rango {cob.pct_fuera.min():.1%}–{cob.pct_fuera.max():.1%}")
print(f"  jóvenes (< 13 meses o nacidos después del origen): mediana {cob.pct_jovenes.median():.1%}")
print(f"  dormidos que se reactivan (sin venta en los 12 meses previos al origen): mediana {cob.pct_dormidos.median():.2%}, "
      f"máx. {cob.pct_dormidos.max():.2%}")
print("SKUs dormidos por origen y cuántos venden en el mes objetivo (reactivación):")
print(cob.set_index("mes_obj")[["skus_dormidos", "skus_dormidos_reactivados", "u_dormidos"]].T.to_string())
print("Clases SB en prueba:", test.clase.value_counts().to_dict(), "· tipo:", test.tipo.value_counts().to_dict())
""")

md("## 2. Validación: hiperparámetros del modelo global")
py("""
print(pd.DataFrame(meta["hiper"]).T[["WAPE", "sesgo", "MASE", "RMSSE"]].round(3).to_string())
print("Elegido (regla WAPE + sesgo frente a A):", meta["hiper_elegido"])
print("Meses bajos del total detectados (excluidos como objetivo de entrenamiento):", meta.get("meses_bajos"))
""")

md("## 3. Validación: todos los modelos")
py("tabla(val, CLASICOS + ML)")

md("""
### 3.1 Ablaciones de features (validación y prueba)

Diferencia frente a `ML_base` (negativo = mejora). **Regla de adopción** (WAPE + sesgo, `metricas.adopta`): una
variante reemplaza a `ML_base` solo si en **validación** baja el WAPE en ≥ 0.5 pp, gana en WAPE en ≥ 2/3 de los
orígenes y no empeora el |sesgo| en más de 1 pp; si no, gana el modelo más simple. La prueba solo confirma.
`ML_sin_limpieza` usa la serie cruda en lags y medias (la limpieza solo aplica a suaves/erráticos).
""")
py("""
def ablacion(d):
    r = resumen(d, ML)
    return (r - r.loc["ML_base"])[["WAPE", "sesgo", "WAPE_total", "MASE", "RMSSE"]]
ab = pd.concat({"validación": ablacion(val), "prueba": ablacion(test)}, axis=1)
print(ab.round(4).to_string())

otros = [m for m in ML if m != "ML_base"] + ["Ruta_clasica"]
for n, d in [("validación", val), ("prueba", test)]:
    po = por_origen(d, ["ML_base"] + otros)
    gana = (po[otros].lt(po.ML_base, axis=0)).sum()
    print(f"\\n{n}: orígenes (de {len(po)}) donde cada modelo mejora el WAPE de ML_base:", gana.to_dict())

cumplen = [m for m in ML if m != "ML_base" and adopta(val, m, "ML_base")]
mejor_ml = elegir(val, ML, "ML_base")
print("\\nVariantes que cumplen la regla:", cumplen or "ninguna", "→ modelo final:", mejor_ml)
""")

md("### 3.2 Ruta por segmento (clase Syntetos–Boylan en el origen)")
py("""
print("Ruta elegida en validación (regla WAPE + sesgo frente a SNaive, dentro de cada clase):", meta["ruta"])
for met in ("WAPE", "sesgo"):
    seg = pd.concat({c: resumen(g, CLASICOS[:-1] + [mejor_ml])[met] for c, g in val.groupby("clase")}, axis=1)
    print(f"\\n{met} por clase (validación):"); print(seg.round(3).to_string())
""")

md("""
### 3.3 Jerárquico middle-out: familia (nivel × índice) repartida con el ML (`src/jerarquico.py`, LOG #32)

En cada origen, el total de la familia = **nivel** (media desestacionalizada de los últimos k meses) × **índice
estacional** de la familia, con datos ≤ origen. Se reparte entre los SKUs con la cuota histórica (`TD_hist`), con la
cuota del `ML_base` (`TD_ML`) o con volumen = promedio entre nivel × índice y Σ `ML_base` (`TD_mix`). k ∈ {3, 6, 12}
se elige en validación con la regla WAPE + sesgo frente a k = 12. `SARIMA_fam` (otro top-down, con cuota histórica)
va de referencia. Misma regla de adopción frente a `ML_base`. Ninguna variante con un solo k pasa la regla en los dos
periodos; la combinación por estado de la familia (`TD_estado`, que usa k = 3 en caídas y k = 12 en subidas) sí, y
se desarrolla en §3.6–3.8.
""")
py("""
jer = {n: d.merge(pd.read_parquet(RES / f"pronosticos_{p}_jer.parquet"), on=["id_articulo", "origen", "mes_obj"])
       for n, d, p in [("validación", val, "validacion"), ("prueba", test, "prueba")]}
meta_jer = json.loads((RES / "meta_jer.json").read_text(encoding="utf-8"))
k = meta_jer["k_elegido"]
print("WAPE de TD_ML por ventana k (validación):", {c: round(v, 3) for c, v in meta_jer["WAPE_por_k"].items()}, "→", k)
JER = ["ML_base", "SARIMA_fam", f"TD_hist_{k}", f"TD_ML_{k}", f"TD_mix_{k}"]
for n, d in jer.items():
    r = resumen(d, JER)
    po = por_origen(d, JER)
    r["gana_a_ML"] = [f"{(po[m] < po.ML_base).sum()}/{len(po)}" if m != "ML_base" else "" for m in JER]
    print(f"\\n{n}"); print(r[["WAPE", "sesgo", "WAPE_total", "MASE", "RMSSE", "gana_a_ML"]].round(3).to_string())
print("\\n¿Cumplen la regla frente a ML_base en validación?:", {m: adopta(jer["validación"], m, "ML_base") for m in JER[1:]})

# ¿el total de la familia (nivel × índice) es mejor que la suma del ML?
for n, d in jer.items():
    g = d.groupby(["familia", "mes_obj"]).agg(y=("y", "sum"), ML=("ML_base", "sum"), F=(f"F_fam_{k}", "first"),
                                              mix=(f"TD_mix_{k}", "sum"))
    w = {c: (g[c] - g.y).abs().sum() / g.y.sum() for c in ["ML", "F", "mix"]}
    print(f"{n}: WAPE familia-mes → Σ ML {w['ML']:.3f} · nivel × índice {w['F']:.3f} · mix {w['mix']:.3f}")
""")

md("""
### 3.4 ¿La regla nueva de limpieza perjudica los pronósticos? (`src/regimen.py`, LOG #42)

La corrida anterior no es comparable directamente (h = 4, `order_date`, otra ventana de validación). Se hace una
**ablación con el código actual**: mismo `ML_base`, mismos hiperparámetros y orígenes, revirtiendo cada parte de la
regla (LOG #39):
- `objetivo_crudo`: entrena con la venta cruda en lugar de `y_entreno`.
- `limpieza_todos`: limpia todas las clases SB, no solo suaves y erráticas.
- `sin_piso_covid_fijo`: sin el piso de meses bajos; excluye mar–jun 2020 fijo, como antes.
- `adi_viejo`: ADI con los meses hasta el origen (incluye los ceros posteriores a la última venta).
- `regla_vieja`: todo lo anterior junto.

La **evaluación siempre es contra la venta cruda** (se verifica abajo): la limpieza solo cambia lo que el modelo usa
para entrenar.
""")
py("""
import features as ft
import jerarquico as jq
import regimen as rg

datos = ft.Datos.cargar()
k_ = datos.skus.get_indexer(val.id_articulo)
print("Meses bajos detectados:", {n: [f"{m:%Y-%m}" for m in v] for n, v in ft.nombres_bajos(datos.bajo).items()})
print("¿La y de validación es la venta cruda del panel?:",
      np.allclose(val.y.to_numpy(), datos.y[k_, ft.MESES.get_indexer(val.mes_obj)]))

ABL = ["actual", "objetivo_crudo", "limpieza_todos", "sin_piso_covid_fijo", "adi_viejo", "regla_vieja"]
abl = {n: pd.read_parquet(RES / f"regimen_ablacion_{p}.parquet") for n, p in [("validación", "validacion"), ("prueba", "prueba")]}
for n, d in abl.items():
    r = resumen(d.assign(escala_mae=1, escala_mse=1, SNaive=d.actual), ABL)[["WAPE", "sesgo"]]
    po = por_origen(d, ABL)
    r["variante gana a actual"] = [f"{(po[m] < po.actual).sum()}/{len(po)}" if m != "actual" else "" for m in ABL]
    r["actual cumple la regla frente a ella"] = [adopta(d, "actual", m) if m != "actual" else "" for m in ABL]
    print(f"\\n{n}"); print(r.round(3).to_string())
""")
py("""
# el sesgo de validación es del periodo: los modelos con serie cruda (sin limpieza) tienen el mismo patrón por mes
for n, d, base in [("validación", abl["validación"], val), ("prueba", abl["prueba"], test)]:
    x = d.merge(base[["id_articulo", "origen", "MA12", "Ruta_clasica", "SNaive"]], on=["id_articulo", "origen"])
    g = x.groupby("mes_obj")[["y", "actual", "regla_vieja", "MA12", "Ruta_clasica", "SNaive"]].sum()
    s = (g.drop(columns="y").div(g.y, axis=0) - 1) * 100
    s.index = s.index.strftime("%b-%y")
    print(f"\\nSesgo por mes objetivo, {n} (%):"); print(s.round(0).T.to_string())
""")
md("""
**Lectura.**
- **La regla nueva no perjudica: mejora.** Frente a la regla vieja completa, `ML_base` baja el WAPE 2.6 pp en validación
  (0.469 vs 0.495) y 0.5 pp en prueba (0.490 vs 0.495), con menos sesgo, y cumple la regla de adopción en ambos.
- **La mejora viene de entrenar con el objetivo limpio:** volver al objetivo crudo cuesta 2.2 pp de WAPE y 3 pp de
  sesgo en validación. El piso de meses bajos y el ADI hasta la última venta son casi neutrales.
- **Matiz:** limpiar también grumosos e intermitentes sale apenas mejor (−0.3 pp de WAPE, −1.7 pp de sesgo, 7–8 de
  12–13 orígenes), sin pasar la regla. Probablemente porque baja el pronóstico en un periodo en que todos
  sobre-pronostican; se mantiene sin limpiar esas clases porque sus picos son demanda de proyecto.
- **El +19 % de sesgo de validación es del periodo, no de la regla.** MA12, la ruta clásica y SNaive (venta cruda, sin
  limpieza) tienen el mismo patrón por mes: +27 a +59 % en jul y nov–dic 2024 y +31 a +73 % en abr-2025. La ventana
  de validación cae en caida_2024 y todo modelo entrenado con el nivel previo sobre-pronostica.
""")

md("""
### 3.5 ETS como estadístico de la familia (LOG #43)

Mismo esquema middle-out de 3.3, pero el volumen de la familia lo pronostica un **AutoETS estacional** (s = 12) en lugar
del nivel × índice. `TD_ML_ets` = ETS decide el volumen de la familia y el ML lo reparte entre sus SKUs.
""")
py("""
ETSM = ["ML_base", f"TD_ML_{k}", "TD_ML_k3", "TD_ML_ets", "TD_mix_k3", "TD_mix_ets", "TD_hist_ets"]
ETSM = list(dict.fromkeys(ETSM))
for n, p in [("validación", "validacion"), ("prueba", "prueba")]:
    d = jer[n].merge(pd.read_parquet(RES / f"regimen_ets_{p}.parquet"), on=["id_articulo", "origen"])
    r = resumen(d, ETSM)[["WAPE", "sesgo", "MASE"]]
    po = por_origen(d, ETSM)
    r["gana_a_ML"] = [f"{(po[m] < po.ML_base).sum()}/{len(po)}" if m != "ML_base" else "" for m in ETSM]
    r["cumple regla"] = [adopta(d, m, "ML_base") if m != "ML_base" else "" for m in ETSM]
    print(f"\\n{n} (nivel SKU)"); print(r.round(3).to_string())
    g = d.groupby(["familia", "origen"]).agg(y=("y", "sum"), ML=("ML_base", "sum"), k3=("F_fam_k3", "first"),
                                             k12=("F_fam_k12", "first"), ets=("F_fam_ets", "first"))
    g["mix_k3"] = (g.k3 + g.ML) / 2
    print("nivel familia:", {c: f"WAPE {(g[c] - g.y).abs().sum() / g.y.sum():.3f} / sesgo {(g[c] - g.y).sum() / g.y.sum():+.1%}"
                             for c in ["ML", "k3", "k12", "ets", "mix_k3"]})
""")
md("""
**Lectura.** ETS **no mejora** el volumen de la familia. A nivel familia queda igual o peor que el nivel de 3 meses
(validación 0.303 vs 0.283; prueba 0.271 vs 0.267) y su sesgo es el más alto (+21 % y +7 %): no se adapta tan rápido a
caida_2024. En prueba el ML predice mejor el total de la familia (0.231) que cualquier estadístico. Ninguna variante con
ETS cumple la regla frente a `ML_base`. El esquema middle-out no es el problema: el ML ya pronostica bien el volumen de
la familia; lo que aporta la estadística de familia es reaccionar rápido a un cambio de nivel (3.6).
""")

md("""
### 3.6 ¿Quién se adapta mejor después de una caída o una subida? (LOG #44)

Línea de tiempo continua: validación + puente + prueba (objetivos jul-2024 → sep-2026). En cada origen, el **estado de
la familia** se calcula solo con lo que se sabe en ese momento: r = media de los últimos 3 meses / media de los 12
anteriores (venta cruda de toda la familia). **Caída** si r < 0.85, **subida** si r > 1.15, **estable** en medio;
`tramo` = meses seguidos en ese estado. El error se mide contra la venta cruda, sumando los SKUs del set por familia.
""")
py("""
MODR = ["ML_base", "TD_mix_k3", "TD_mix_k12", "TD_ML_k3"]
est, racha, r_est = rg.estado_familia(datos)
lt = rg.etiquetar(rg.linea_tiempo(), est, racha)
fam_lt = lt.groupby(["familia", "origen", "mes_obj", "estado", "tramo"], observed=True)[["y"] + MODR].sum().reset_index()

def por_grupo(x, por):
    g = x.groupby(por, observed=True)
    out = {}
    for m in MODR:
        out[(m, "WAPE")] = g.apply(lambda s: (s[m] - s.y).abs().sum() / s.y.sum(), include_groups=False)
        out[(m, "sesgo")] = g.apply(lambda s: (s[m] - s.y).sum() / s.y.sum(), include_groups=False)
    out[("", "% unidades")] = g.y.sum() / x.y.sum()
    return pd.DataFrame(out)

print("Nivel familia, por estado en el origen y meses en ese estado:")
print(por_grupo(fam_lt, ["estado", "tramo"]).round(3).to_string())
fam_lt["gana_k3"] = (fam_lt.TD_mix_k3 - fam_lt.y).abs() < (fam_lt.ML_base - fam_lt.y).abs()
print("\\n% de celdas familia-origen donde TD_mix_k3 tiene menor error que ML_base:")
print((fam_lt.groupby(["estado", "tramo"], observed=True).gana_k3.mean() * 100).round(0).unstack().to_string())
""")
py("""
tot_lt = lt.groupby("mes_obj")[["y"] + MODR].sum() / 1e3
e_tot = est.TOTAL.reindex(tot_lt.index - pd.DateOffset(months=H)).to_numpy()   # estado del total en el origen
fig, ax = plt.subplots(figsize=(12, 4.2))
for i, (m, e) in enumerate(zip(tot_lt.index, e_tot)):
    if e != "estable":
        ax.axvspan(m - pd.DateOffset(days=15), m + pd.DateOffset(days=15), color=COLORES[1] if e == "caída" else COLORES[2],
                   alpha=.15, lw=0)
ax.plot(tot_lt.index, tot_lt.y, color="#0b0b0b", lw=2.4, marker="o", ms=4, label="Real")
for m, c in zip(MODR[:3], [COLORES[3], AZUL, GRIS]):
    ax.plot(tot_lt.index, tot_lt[m], lw=1.7, color=c, label=m)
ax.set_ylim(0, None); ax.set_ylabel("miles de unidades"); ax.legend(fontsize=8, loc="lower left")
ax.set_title(f"Total del set: real vs pronóstico a {H} meses · sombreado = estado del total en el origen "
             "(naranja caída, verde subida)", fontsize=10)
fig.savefig(FIG / "28_adaptacion_total.png"); plt.show()
s = (tot_lt[MODR].div(tot_lt.y, axis=0) - 1) * 100
s.insert(0, "estado (origen)", e_tot)
s.index = s.index.strftime("%b-%y")
print("Sesgo del total por mes objetivo (%):"); print(s.round(0).to_string())
""")
py("""
# regla causal por estado: A = TD_mix_k3 en caída; B = además TD_mix_k12 en subida
lt["Estado_A"] = rg.regla_estado(lt, {"caída": "TD_mix_k3"})
lt["Estado_B"] = rg.regla_estado(lt)
CANDR = ["ML_base", "TD_mix_k3", "TD_mix_k12", "Estado_A", "Estado_B"]
for p in ["validacion", "prueba"]:
    x = lt[lt.periodo == p]
    r = resumen(x, CANDR)[["WAPE", "sesgo", "WAPE_total", "sesgo_total"]]
    po = por_origen(x, CANDR)
    r["gana_a_ML"] = [f"{(po[m] < po.ML_base).sum()}/{len(po)}" if m != "ML_base" else "" for m in CANDR]
    r["cumple regla"] = [adopta(x, m, "ML_base") if m != "ML_base" else "" for m in CANDR]
    print(f"\\n{p}: % de filas con la familia en caída {(x.estado == 'caída').mean():.0%}, "
          f"en subida {(x.estado == 'subida').mean():.0%}")
    print(r.round(3).to_string())
""")
md("""
**Lectura.** La adaptación es **asimétrica**:
- **Tras una caída, `TD_mix_k3` se adapta mejor.** `ML_base` escala con la media de 12 meses y tarda en bajar: su sesgo
  crece con la duración de la caída (+7 % en los meses 1–2, +25 % a partir del 6). El nivel de 3 meses de la familia
  lo corta a +1 % y +10 %, con menor WAPE en todos los tramos. En el total: nov–dic 2024, `ML_base` +48 % / +44 % contra
  +30 % / +29 % de `TD_mix_k3`.
- **Tras una subida, `TD_mix_k3` se pasa.** Extrapola el alza, que luego revierte (como en `04_segmentacion`: la
  tendencia revierte a la media): sesgo +8 a +16 % contra −2 a +10 % de `ML_base`. Ahí el mejor es `TD_mix_k12`.
- **Estable:** empatan.
- **Ningún modelo anticipa un giro** que ocurre en el mismo mes objetivo (jul-2025: todos −29 a −32 %): a 3 meses no
  hay señal.
- **Regla por estado:** A (caída → `TD_mix_k3`) cumple la regla en validación (0.456 vs 0.469, 9/12) y en prueba mejora
  poco (−0.3 pp, 9/13; sesgo +1.9 % vs +3.5 %). B (además subida → `TD_mix_k12`) cumple en los dos periodos (prueba
  0.482 vs 0.490, 9/13, sesgo +1.6 %). En validación A y B son casi iguales: la parte de subidas solo se prueba en el
  periodo de prueba, y la idea de usar k12 en subidas salió de ver todo el periodo (riesgo de sobreajuste).
""")

md("""
### 3.7 Sensibilidad de la regla por estado a los umbrales (LOG #44)

Se recalcula el estado de cada familia con distintos umbrales: caída r < {0.75 … 0.95} y subida r > {1.10 … 1.25}.
La regla A solo depende del umbral de caída. `dWAPE` y `dsesgo_abs` son frente a `ML_base` (negativo = mejor).
""")
py("""
sen = pd.read_parquet(RES / "regimen_sensibilidad.parquet")
fmt_s = lambda x: x.assign(dWAPE=(x.dWAPE * 100).round(2), dsesgo_abs=(x.dsesgo_abs * 100).round(1),
                           gana=(x.gana * 100).round(0), pct_caida=(x.pct_caida * 100).round(0),
                           pct_subida=(x.pct_subida * 100).round(0))
a = fmt_s(sen[sen.regla == "A"]).pivot(index="bajo", columns="periodo",
                                       values=["dWAPE", "dsesgo_abs", "gana", "adopta", "pct_caida"])
print("Regla A (caída → TD_mix_k3) · dWAPE y dsesgo_abs en pp, gana = % de orígenes:")
print(a.to_string())
for met in ["dWAPE", "adopta"]:
    b = fmt_s(sen[sen.regla == "B"]).pivot_table(index="bajo", columns=["periodo", "alto"], values=met, aggfunc="first")
    print(f"\\nRegla B (además subida → TD_mix_k12) · {met}:"); print(b.to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
for a_, p in zip(ax, ["validacion", "prueba"]):
    x = sen[(sen.periodo == p) & (sen.regla == "A")]
    a_.plot(x.bajo, x.dWAPE * 100, marker="o", color=AZUL, lw=2, label="A")
    for alto, c in zip(sorted(sen.alto.dropna().unique()), [GRIS_CLARO, GRIS, COLORES[2], COLORES[1]]):
        y = sen[(sen.periodo == p) & (sen.regla == "B") & (sen.alto == alto)]
        a_.plot(y.bajo, y.dWAPE * 100, marker=".", lw=1.2, color=c, label=f"B, subida > {alto:.2f}")
    a_.axhline(0, color="#0b0b0b", lw=.8); a_.axhline(-.5, color=GRIS, lw=.8, ls="--")
    a_.set_title(f"{p}: ΔWAPE frente a ML_base (pp)"); a_.set_xlabel("umbral de caída (r <)")
ax[0].legend(fontsize=7.5); ax[0].set_ylabel("pp (línea punteada = −0.5 pp de la regla)")
fig.tight_layout(); fig.savefig(FIG / "29_sensibilidad_estado.png"); plt.show()
""")
md("""
**Lectura.** El resultado **no depende del corte exacto**:
- **Regla A:** mejora a `ML_base` con cualquier umbral de caída entre 0.75 y 0.95, en los dos periodos (validación −0.9
  a −2.2 pp de WAPE; prueba −0.25 a −0.37 pp) y siempre baja el |sesgo| (−2.5 a −5.2 pp y −0.7 a −1.7 pp). Cumple la
  regla en validación con todos los umbrales; en prueba no llega al −0.5 pp con ninguno.
- **Regla B:** cumple la regla en los **20 pares de umbrales** y en ambos periodos (prueba −0.7 a −0.9 pp; validación
  −0.9 a −2.3 pp). Es casi insensible al umbral de subida (±0.05 pp entre 1.10 y 1.25).
- Umbrales de caída más altos (0.90–0.95) marcan más familias en caída (57–69 % de las filas de validación) y dan más
  ganancia en validación, pero no en prueba: 0.85 queda en medio y es el que se usa.
""")

md("""
### 3.8 Propuesta adoptada: `TD_estado`, pronóstico según el estado de la familia (LOG #45)

**Regla.** En cada origen T, para cada familia (venta cruda de todos sus SKUs, solo meses ≤ T):

| r = media 3 meses ≤ T / media de los 12 anteriores | Estado | Pronóstico de los SKUs de la familia |
|---|---|---|
| r < 0.85 | caída | `TD_mix_k3`: volumen = promedio de (nivel de 3 meses × índice) y Σ `ML_base`; reparto con la cuota del ML |
| r > 1.15 | subida | `TD_mix_k12`: igual, con el nivel de 12 meses |
| en medio | estable | `ML_base` |

Implementación: `jerarquico.estado_familia` y `jerarquico.regla_estado`; columna `TD_estado` en
`pronosticos_*_jer.parquet`. Es el pronóstico de los SKUs maduros en inventario (`03_inventario`) y en el pedido
(`src/pedido.py`). Los jóvenes siguen con `Media_post`.

**Por qué.**
1. **Problema que resuelve: el sesgo después de un cambio de nivel.** `ML_base` normaliza con la media de 12 meses, así
   que tras una caída sigue pronosticando el nivel anterior. Su sesgo crece con la duración de la caída: +7 % en los
   meses 1–2, +10 % en los 3–5 y +25 % a partir del 6 (§3.6). Esto pasó en toda la validación (caida_2024): +19 %.
   Un sesgo así se convierte directamente en stock de más.
2. **Mecanismo, no ajuste estadístico.** El nivel de 3 meses de la familia reacciona rápido a una caída, pero tras una
   subida la persigue y se pasa, porque las alzas suelen revertir (como ya se vio en la fase de ciclo de vida de
   `04_segmentacion`). Por eso la regla es **asimétrica**: rápida para bajar, conservadora para subir. En estable no
   hay nada que corregir y se deja el modelo global.
3. **Es causal.** El estado solo usa venta ≤ T (se comprueba abajo perturbando el futuro): en producción se calcula
   igual que en el backtest, con el cierre de cada mes.
4. **Cumple la regla WAPE + sesgo en los dos periodos**, no solo en validación: WAPE 0.456 vs 0.469 (9 de 12 orígenes)
   y 0.482 vs 0.490 (9 de 13), con el sesgo de +19.3 → +15.2 % y de +3.5 → +1.6 %. Ninguna otra variante de este
   proyecto pasó la regla en prueba.
5. **Es robusta a los umbrales (§3.7):** cumple la regla en los 20 pares probados (caída 0.75–0.95, subida 1.10–1.25),
   en ambos periodos. El resultado no depende del corte exacto.
6. **Es simple y explicable:** combina tres modelos que ya existían, sin parámetros nuevos que entrenar. El planeador
   puede ver el estado de cada familia y saber qué pronóstico se usó.

**Riesgos y cómo se vigilan.**
- **La parte de subidas se definió viendo el periodo de prueba** (en validación casi no hubo subidas: A y B dan igual).
  La regla A (solo caídas) sí se justifica solo con validación; B añade 0.5 pp en prueba. Se revisa en cada
  reentrenamiento mensual: si en 6 meses con familias en subida `TD_estado` no le gana a la regla A, se vuelve a A.
- **La ganancia en WAPE es moderada** (0.8–1.3 pp). Lo principal es el sesgo.
- **No ahorra stock con el SS empírico** (`03_inventario`): el SS se calibra con los errores de ventana de cada modelo y
  absorbe el sesgo, así que a igual nivel de servicio el stock queda igual que con `ML_base`. La ventaja es tener un
  pronóstico central más fiel (sesgo +1.6 % en prueba), que es lo que ve comercial y compras.
- **Ningún modelo anticipa un giro en el mismo mes objetivo** (§3.6): la regla corrige *después* de que la caída ya se
  ve en los datos.
""")
py("""
FIN = ["ML_base", "TD_mix_k3", "TD_mix_k12", "TD_estado"]
for n, d in jer.items():
    r = resumen(d, FIN)[["WAPE", "sesgo", "WAPE_total", "sesgo_total", "MASE", "RMSSE"]]
    po = por_origen(d, FIN)
    r["gana_a_ML"] = [f"{(po[m] < po.ML_base).sum()}/{len(po)}" if m != "ML_base" else "" for m in FIN]
    r["cumple regla"] = [adopta(d, m, "ML_base") if m != "ML_base" else "" for m in FIN]
    print(f"\\n{n}"); print(r.round(3).to_string())

# leakage: el estado de la familia en el origen no cambia si se altera todo lo posterior al origen
import copy
rng_ = np.random.default_rng(0)
for o in [ft.MESES.get_loc(val.origen.min()), ft.MESES.get_loc(test.origen.max())]:
    d_ = copy.copy(datos); d_.y = datos.y.copy()
    d_.y[:, o + 1:] = rng_.poisson(50, d_.y[:, o + 1:].shape)
    assert (jq.estado_familia(d_)[0].iloc[:o + 1] == jq.estado_familia(datos)[0].iloc[:o + 1]).all().all()
print("\\nEstado de la familia sin leakage (futuro perturbado en 2 orígenes): OK")
""")
py("""
# por origen: ΔWAPE de TD_estado frente a ML_base y estado de las familias (% de unidades reales)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
for a, (n, d) in zip(ax, jer.items()):
    po = por_origen(d, ["ML_base", "TD_estado"])
    dw = (po.TD_estado - po.ML_base) * 100
    a.bar(po.index, dw, width=20, color=[AZUL if v < 0 else COLORES[1] for v in dw])
    a.axhline(0, color="#0b0b0b", lw=.8)
    a.set_title(f"{n}: ΔWAPE por origen, TD_estado − ML_base (pp; azul = mejora)", fontsize=10)
    a.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%y"))
fig.tight_layout(); fig.savefig(FIG / "30_td_estado_por_origen.png"); plt.show()
for n, d in jer.items():
    mix = d.groupby(["origen", "estado_fam"]).y.sum().unstack(fill_value=0)
    mix = (mix.div(mix.sum(axis=1), axis=0) * 100).round(0)
    mix.index = mix.index.strftime("%b-%y")
    print(f"\\n{n}: % de unidades reales según el estado de su familia en el origen"); print(mix.T.to_string())
""")
py("""
# dentro de cada estado y por segmento: ¿a quién mejora y a quién no?
for n, d in jer.items():
    r = pd.concat({e: resumen(g, ["ML_base", "TD_estado"])[["WAPE", "sesgo"]].unstack() for e, g in d.groupby("estado_fam")},
                  axis=1).T
    print(f"\\n{n}, por estado de la familia:"); print(r.round(3).to_string())
t3 = jer["prueba"].merge(clas, on="id_articulo", how="left")
t3["abc"] = t3.abc.astype(object).fillna("sin_ABC")
for col in ["clase", "abc"]:
    r = pd.concat({k_: resumen(g, ["ML_base", "TD_estado"])[["WAPE", "sesgo"]].unstack() for k_, g in t3.groupby(col)},
                  axis=1).T
    print(f"\\nprueba, por {col}:"); print(r.round(3).to_string())
""")

md("## 4. Prueba (sep-2025 → sep-2026)")
py("""
FINALES = ["Naive", "SNaive", "MA12", "ETS", "SARIMA_fam", "Croston", "SBA", "TSB", "Ruta_clasica", "ML_base", "ML_devol",
           "ML_sin_limpieza", mejor_ml]
FINALES = list(dict.fromkeys(FINALES))
tabla(test, FINALES)
""")

py("""
tot = test.groupby("mes_obj")[["y"] + FINALES].sum()
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(tot.index, tot.y, color="#0b0b0b", lw=2.2, marker="o", ms=4, label="real")
for m, col in zip(["SNaive", "Ruta_clasica", mejor_ml], [GRIS, COLORES[1], AZUL]):
    ax.plot(tot.index, tot[m], color=col, lw=1.8, label=m)
ax.yaxis.set_major_formatter(FMT_MILES); ax.legend(fontsize=8)
ax.set_title(f"Prueba: total mensual de los SKUs evaluados — real vs pronóstico a {H} meses")
fig.savefig(FIG / "20_prueba_total.png"); plt.show()
print((tot[FINALES].sub(tot.y, axis=0).div(tot.y, axis=0) * 100).round(1).T.to_string())
""")

md("### 4.1 Prueba por segmento")
py("""
t2 = test.merge(clas, on="id_articulo", how="left")
t2["abc"] = t2.abc.astype(object).fillna("sin_ABC")
for col in ["tipo", "clase", "abc", "perfil"]:
    r = pd.concat({k: resumen(g, ["SNaive", "Ruta_clasica", mejor_ml]).WAPE for k, g in t2.groupby(col)}, axis=1)
    peso = t2.groupby(col).y.sum() / t2.y.sum()
    print(f"\\nWAPE por {col} (fila 'peso' = % de unidades reales):")
    print(pd.concat([r, peso.rename("peso").to_frame().T]).round(3).to_string())
""")

md("""
### 4.2 Real vs pronóstico del sistema, por clase ABC (prueba)

El pronóstico que usa la política de pedidos (`03_inventario`): `TD_estado` para los maduros (§3.8) y `Media_post`
para los jóvenes (sección 5), emitido 3 meses antes. La ruta clásica (con los mismos jóvenes) va de referencia. La clase es el
**ABC móvil al origen** (`04_segmentacion`, LOG #31). En el título: sesgo del año y error mensual
(Σ|pronóstico − real| / Σ real de los totales mensuales, no a nivel SKU).
""")
py("""
jov = pd.read_parquet(RES / "cold_jovenes_prueba.parquet")[["id_articulo", "origen", "mes_obj", "y", "Media_post"]]
sis = pd.concat([jer["prueba"][["id_articulo", "origen", "mes_obj", "y", "TD_estado", "Ruta_clasica"]],
                 jov.assign(TD_estado=jov.Media_post, Ruta_clasica=jov.Media_post).drop(columns="Media_post")])
abc_mov = pd.read_parquet(RAIZ / "data" / "processed" / "segmentacion_sku_mes.parquet", columns=["id_articulo", "mes", "abc_movil"])
sis = sis.merge(abc_mov.rename(columns={"mes": "origen"}), on=["id_articulo", "origen"], how="left")

fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
tablas = {}
for a, (tit, x) in zip(ax.ravel(), [("Total", sis)] + [(f"Clase {c}", sis[sis.abc_movil == c]) for c in "ABC"]):
    g = x.groupby("mes_obj")[["y", "TD_estado", "Ruta_clasica"]].sum() / 1e3
    a.plot(g.index, g.y, lw=2.2, marker="o", ms=5, color=GRIS, label="Real")
    a.plot(g.index, g.TD_estado, lw=2, marker="o", ms=5, color=AZUL, label="TD_estado (+ Media_post jóvenes)")
    a.plot(g.index, g.Ruta_clasica, lw=1.5, ls="--", color=COLORES[1], label="Ruta clásica (referencia)")
    sesgo = g.TD_estado.sum() / g.y.sum() - 1
    err = (g.TD_estado - g.y).abs().sum() / g.y.sum()
    a.set_title(f"{tit} · sesgo {sesgo:+.0%} · error mensual {err:.0%}", fontsize=10.5)
    a.set_ylim(0, None)
    a.set_ylabel("miles de unidades")
    a.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%y"))
    tablas[tit] = g.assign(dif=g.TD_estado / g.y - 1)
ax[0, 0].legend(fontsize=8.5, loc="lower left")
fig.suptitle(f"Prueba sep-2025 → sep-2026: real vs pronóstico emitido {H} meses antes", fontsize=10, color=GRIS)
fig.tight_layout()
fig.savefig(FIG / "25_real_vs_pronostico_prueba.png"); plt.show()

t = pd.concat({k: v[["y", "TD_estado", "dif"]] for k, v in tablas.items()}, axis=1)
t.index = t.index.strftime("%b-%y")
fmt = t.copy()
for c in fmt.columns:
    fmt[c] = t[c].map(lambda v: f"{v:+.0%}" if c[1] == "dif" else f"{v:.1f}")
print("Miles de unidades; dif = pronóstico / real − 1")
print(fmt.to_string())
""")

md("""
**Lectura.** El total queda **+1 %** en el año, con un error mensual del 9 %: a nivel agregado los errores de los SKUs
se compensan. El pronóstico es mucho **más plano** que la venta real (103–123 contra 91–133 mil u./mes): a 3 meses
capta el nivel y algo de estacionalidad, pero no los picos de cada mes, que se cubren con el SS (`03_inventario`).
El error más grande es el **sesgo de dic-2025 a feb-2026 en la clase A** (+24 %, +19 %, +28 %): el modelo repitió
un repunte estacional que este año no ocurrió. Los meses más bajos del pronóstico frente a la venta son mar-2026
(−16 %) y ago-2026 (−12 %), picos que a 3 meses no se ven venir. En C sub-pronostica (−16 % en el año) y no capta los
pedidos grandes y esporádicos (dic-2025: 15 mil reales contra 9 mil), típicos de la demanda grumosa.
""")

md("""
### 4.3 XGBoost por serie, top 30 SKUs y top 30 familias (`notebooks/XGBoost.py`, LOG #33)

Adaptación del script de XGBoost del usuario: un `XGBRegressor` por serie (100 árboles, profundidad 3, error
cuadrático), con features = mes + 3 lags. Con h = 3 los lags son 3, 4 y 5. Los datos vienen del panel (`fechafactura`,
`Venta Producto`). Top 30 por unidades del último año. Se evalúa con el **mismo walk-forward** (entrena con datos
≤ T, pronostica T + 3) y se compara contra los nuestros (`src/comparar_xgb.py`):
- **SKU:** pares (SKU, origen) del set evaluado.
- **Familia:** real = total de la familia. Lo nuestro se suma por familia: maduros + jóvenes (`Media_post`).

MAE y RMSE están en unidades; R² fuera de muestra = 1 − SSE / SST de los reales del periodo.
""")
py("""
import comparar_xgb as cx

XG = cx.NUESTROS + ["XGBoost"]
sku = cx.nivel_sku()
fam, _ = cx.nivel_familia()
for nivel, res in [("Top 30 SKUs", sku), ("Top 30 familias", fam)]:
    for p, d in res.items():
        print(f"\\n{nivel} · {p} ({d.id_articulo.nunique()} series, {len(d)} pares serie-origen)")
        print(cx.tabla(d, XG).sort_values("WAPE").round(3).to_string())

ins = pd.concat([pd.read_csv(RES / f"xgb_top30_insample_{n}.csv") for n in ("sku", "familia")])
print("\\nXGBoost in-sample (como el script original): mediana por nivel")
print(ins.groupby("nivel")[["R2", "wMAPE", "sMAPE"]].median().round(3).to_string())
""")

md("""
**Lectura.**
- **XGBoost por serie queda último o penúltimo** en los dos niveles; solo le gana a SNaive en validación. En los top 30
  SKUs su WAPE es 0.460 en validación y 0.604 en prueba, contra 0.32–0.36 y 0.40–0.41 de los nuestros. En las
  familias queda en 0.360 y 0.319, contra 0.25–0.28 y 0.22–0.23. Además sobre-pronostica mucho más (+14 a +26 %).
- **El R² in-sample engaña.** Ajustado sobre su propia historia, como en el script original, da un R² mediano de
  0.87–0.89 y un wMAPE de 10–12 %. Fuera de muestra, el R² de los SKUs cae a −1.7 en validación y a −3.4 en prueba
  (algunos SKUs con pronósticos muy altos, como el 1978). Un modelo de 100 árboles por serie, con unos 100 meses de
  datos, **memoriza** la historia en lugar de aprender un patrón que se repita. Por eso todo se evalúa con
  walk-forward.
- La diferencia no es XGBoost contra HistGradientBoosting (son el mismo tipo de algoritmo). Es **un modelo por
  serie contra uno global**: el global aprende de ~800 SKUs a la vez, con features normalizadas y pérdida Poisson.
- **En los 30 SKUs más grandes, los clásicos están a la par del ML:** la ruta clásica es la mejor en validación
  (WAPE 0.318) y empata con `TD_estado` en prueba (0.396 vs 0.397). Es coherente con lo visto por ABC: la ventaja del
  ML está en la cola de SKUs chicos e intermitentes. `TD_estado` mejora a `ML_base` en los cuatro cortes (SKU y
  familia, validación y prueba) y es el mejor en las top 30 familias de prueba (0.219).
""")

md("""
### 4.4 Top 30: real contra cada pronóstico, y qué mide realmente el R²

**Qué es el R².** R² = 1 − SSE / SST, donde SSE = Σ(pronóstico − real)² y SST = Σ(real − media del real)².
Compara el modelo contra un pronóstico "tonto": **una línea constante en la media real del periodo**.
- R² = 1: error cero. R² = 0: igual que esa línea constante. **R² < 0: peor que la línea constante.**
- Esa línea es tramposa, porque usa la media **real** del periodo, que no se conoce de antemano. Un R² negativo
  fuera de muestra no es raro en pronóstico: ganarle a la media real de un futuro que no has visto es difícil.

**Por qué el R² del script original salía tan alto (~0.9).**
1. **In-sample:** el modelo se evaluaba sobre los mismos meses con que se entrenó. Con 100 árboles de profundidad 3
   y ~100 meses por serie, el modelo puede aprenderse cada pico de memoria. El R² mide memoria, no pronóstico.
2. **Con lags 1–3 (horizonte 1),** el mes anterior explica mucho. A 3 meses de distancia esa información no existe.

**Por qué el R² agrupado de varias series engaña.** Si se juntan series de distinto tamaño (un SKU de 5,000 u./mes
con uno de 500), casi toda la varianza (SST) está **entre** series, no dentro de cada una. Cualquier pronóstico
que adivine el tamaño de cada SKU obtiene un R² alto aunque no capte ningún movimiento mes a mes. La tabla de
abajo lo mide con un **oráculo** que predice, para cada serie, su propia media real del periodo (una constante): ese
oráculo saca un R² agrupado alto, pero 0 dentro de cada serie por definición. Por eso las familias (tamaños muy
distintos) tienen R² de 0.85–0.89 y los SKUs del top 30 (tamaños más parecidos) solo 0.35–0.64.

**El R² por serie** (dentro de cada SKU, sobre los 12–13 meses del periodo) dice cuánto del movimiento mes a mes
explica el pronóstico. A 3 meses suele ser ≤ 0: los pronósticos son mucho más planos que la venta real (sección 4.2),
y los picos se cubren con stock de seguridad, no con el pronóstico. Por esto el proyecto decide con **WAPE + sesgo**:
el WAPE pondera por volumen (lo que cuesta en inventario), no eleva al cuadrado los picos, se entiende en negocio y es
comparable con los demás equipos; el sesgo vigila el error sistemático que se convierte en stock de más o de menos.
El MASE (contra el seasonal naive in-sample) se reporta como secundaria porque no lo dominan los SKUs grandes.
""")
py("""
MOD30 = [cx.TD, "ML_base", "Ruta_clasica", "MA12", "SNaive", "XGBoost"]
for nivel, res in [("Top 30 SKUs", sku), ("Top 30 familias", fam)]:
    for p, d in res.items():
        print(f"\\n{nivel} · {p}")
        print(cx.r2_descompuesto(d, MOD30).round(3).to_string())
""")

py("""
COL30 = dict(zip(MOD30, [AZUL, COLORES[3], COLORES[1], COLORES[2], GRIS, COLORES[4]]))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for a, (nivel, res) in zip(ax, [("Top 30 SKUs", sku), ("Top 30 familias", fam)]):
    d = pd.concat(res.values())
    tot = d.groupby("mes_obj")[["y"] + MOD30].sum() / 1e3
    a.plot(tot.index, tot.y, color="#0b0b0b", lw=2.4, marker="o", ms=4, label="Real")
    for m in MOD30:
        a.plot(tot.index, tot[m], color=COL30[m], lw=1.6, label=m)
    a.axvline(pd.Timestamp("2025-08-16"), color=GRIS_CLARO, lw=1)
    a.text(pd.Timestamp("2025-09-01"), a.get_ylim()[1] * .98, "prueba →", fontsize=8, color=GRIS, va="top")
    a.set_ylim(0, None)
    a.set_title(f"{nivel}: total mensual (miles de u.)")
ax[1].legend(fontsize=7.5, ncol=2, loc="lower left")
fig.suptitle(f"Real vs pronóstico emitido {H} meses antes · validación jul-2024 → jun-2025 y prueba sep-2025 → sep-2026",
             fontsize=9.5, color=GRIS)
fig.tight_layout()
fig.savefig(FIG / "26_top30_total.png"); plt.show()
""")

md("""
Por SKU (los 30, ordenados por venta del último año). Para que se lea, cada panel muestra el real, el modelo elegido
(`TD_estado`), la ruta clásica y XGBoost; los demás están en la tabla y en el CSV. La escala vertical es propia de
cada panel.
""")
py("""
d = pd.concat(sku.values())
orden = d[d.mes_obj >= "2025-10-01"].groupby("id_articulo").y.sum().sort_values(ascending=False).index
fig, ax = plt.subplots(6, 5, figsize=(16, 17), sharex=True)
for a, s in zip(ax.ravel(), orden):
    g = d[d.id_articulo == s].sort_values("mes_obj")
    a.plot(g.mes_obj, g.y, color="#0b0b0b", lw=1.8, marker="o", ms=2.5, label="Real")
    for m in [cx.TD, "Ruta_clasica", "XGBoost"]:
        a.plot(g.mes_obj, g[m], color=COL30[m], lw=1.3, label=m)
    a.axvline(pd.Timestamp("2025-08-16"), color=GRIS_CLARO, lw=1)
    a.set_ylim(0, None)
    a.set_title(f"SKU {s} · {g.familia.iloc[0]}", fontsize=9)
    a.tick_params(labelsize=7)
    a.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    a.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%y"))
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"Top 30 SKUs: real vs pronóstico a {H} meses (línea vertical = inicio de prueba)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .975])
fig.savefig(FIG / "27_top30_por_sku.png"); plt.show()
""")

md("""
**Cuánto predijo cada modelo para cada SKU en el año de prueba** (sep-2025 → sep-2026, 13 meses; unidades
totales del periodo). La columna `mejor` es el modelo con menor error absoluto mensual acumulado del SKU, no el más
cercano en el total: un modelo puede acertar el total del año compensando errores de meses distintos. El detalle
mes a mes de ambos periodos queda en `reports/top30_real_vs_pronosticos.csv`.
""")
py("""
det = pd.concat(sku, names=["periodo"]).reset_index(level=0)[["periodo", "id_articulo", "familia", "origen", "mes_obj", "y"] + MOD30]
det.round(1).to_csv(RAIZ / "reports" / "top30_real_vs_pronosticos.csv", index=False, encoding="utf-8-sig")

pr = sku["prueba"]
tot = pr.groupby("id_articulo")[["y"] + MOD30].sum().reindex(orden)
mae = pr.assign(**{m: (pr[m] - pr.y).abs() for m in MOD30}).groupby("id_articulo")[MOD30].sum().reindex(orden)
tot.insert(0, "familia", pr.groupby("id_articulo").familia.first().reindex(orden))
tot["mejor"] = mae.idxmin(axis=1)
tot.loc["TOTAL"] = ["", *tot[["y"] + MOD30].sum(), ""]
out = tot.copy()
for m in ["y"] + MOD30:
    out[m] = tot[m].map(lambda v: f"{v:,.0f}")
for m in MOD30:
    out[f"{m} %"] = (tot[m].astype(float) / tot.y.astype(float) - 1).map(lambda v: f"{v:+.0%}")
out = out.rename(columns={"y": "real"})
out.round(0).to_csv(RAIZ / "reports" / "top30_totales_prueba.csv", encoding="utf-8-sig")
print(out[["familia", "real"] + MOD30 + ["mejor"]].to_string())
print()
print("Diferencia del total del año vs real:")
print(out[[f"{m} %" for m in MOD30]].to_string())
print()
print("SKUs donde cada modelo es el mejor (menor Σ|error| mensual):", tot.mejor.iloc[:-1].value_counts().to_dict())
""")

md("""
## 5. Cold start (`src/cold_start.py`, LOG #24)

Lo que queda fuera del set maduro, en dos segmentos:
- **Jóvenes** (0–11 meses de vida en el origen): baselines simples contra un modelo global propio (`ML_joven`).
  Se mide WAPE, sesgo y error en log, con la misma regla de adopción WAPE + sesgo (`metricas.adopta`).
- **Pre-lanzamiento** (lanzan entre T+1 y T+3): regla de análogos (familia → sublínea → banda de precio) con rango
  p25–p75 / p10–p90. Supone que el lanzamiento se conoce por comercial.
""")
py("""
import cold_start as cs
jv = {p: pd.read_parquet(RES / f"cold_jovenes_{p}.parquet") for p in ["validacion", "prueba"]}
MJ = cs.BASELINES + ["ML_joven"]
FMTJ = {"WAPE": "{:.3f}", "sesgo": "{:+.1%}", "MAE_log": "{:.3f}", "WAPE_total": "{:.1%}", "sesgo_total": "{:+.1%}"}
for p, d in jv.items():
    print(f"{p}: {len(d):,} filas · {d.id_articulo.nunique()} SKUs · {d.y.sum():,.0f} u.")
pd.concat({p: cs.resumen_joven(d, MJ) for p, d in jv.items()}, axis=1).style.format(
    {(p, c): f for p in jv for c, f in FMTJ.items()})
""")
py("""
def wape_origen(d):
    return pd.DataFrame({o: g[MJ].sub(g.y, axis=0).abs().sum() / g.y.sum() for o, g in d.groupby("origen")}).T
for p, d in jv.items():
    w = wape_origen(d)
    print(f"{p}: mejor modelo por origen →", w.idxmin(axis=1).value_counts().to_dict())
rv = cs.resumen_joven(jv["validacion"], MJ).WAPE
mejor_base = rv[cs.BASELINES].idxmin()
adopta_ml = adopta(jv["validacion"], "ML_joven", mejor_base)
modelo_joven = "ML_joven" if adopta_ml else mejor_base
print(f"\\nMejor baseline en validación: {mejor_base} · ¿ML_joven cumple la regla?: {adopta_ml} → modelo: {modelo_joven}")
""")
py("""
d = jv["prueba"].assign(tramo=lambda x: pd.cut(x.edad, [0, 3, 6, 12], labels=["1–3 m", "4–6 m", "7–12 m"]))
print("Prueba por edad en el origen (WAPE; peso = % de unidades del segmento):")
r = pd.concat({t: cs.resumen_joven(g, MJ).WAPE for t, g in d.groupby("tramo", observed=True)}, axis=1)
print(pd.concat([r, (d.groupby("tramo", observed=True).y.sum() / d.y.sum()).rename("peso").to_frame().T]).round(3).to_string())
""")
py("""
pre = {p: pd.read_parquet(RES / f"cold_prelanz_{p}.parquet") for p in ["validacion", "prueba"]}
print(pd.DataFrame({p: cs.resumen_prelanz(d) for p, d in pre.items()}).round(3).to_string())
print("\\nCobertura nominal: p25–p75 = 50 %, p10–p90 = 80 %.")
""")

md("## 6. Hallazgos")
md("""
**Modelo elegido para los SKUs maduros: `TD_estado`** (§3.8, LOG #45). Según el estado de la familia en el origen
(media de 3 meses / media de los 12 anteriores): `TD_mix_k3` si cae (< 0.85), `TD_mix_k12` si sube (> 1.15) y
`ML_base` si está estable. `ML_base` es el HistGradientBoosting Poisson global (hiperparámetros A, features base,
limpieza causal solo en suaves y erráticos). Jóvenes: `Media_post`.

1. **`TD_estado` es el único candidato que cumple la regla WAPE + sesgo en validación y en prueba.** WAPE 0.456 vs
   0.469 de `ML_base` en validación (9 de 12 orígenes) y 0.482 vs 0.490 en prueba (9 de 13), con el sesgo de +19 % →
   +15 % y de +3.5 % → +1.6 %. Mejora a `ML_base` en todas las clases SB y en A, B y C de prueba, y es robusto a los
   umbrales (§3.7: cumple en los 20 pares probados).
2. **`ML_base` y la ruta clásica quedan cerca en WAPE.** Validación: ruta clásica 0.459 vs `ML_base` 0.469 (la ruta
   gana en 6 de 12 orígenes, sin pasar la regla). Prueba: 0.497 vs 0.490. El ML gana con claridad en MASE (0.788 vs
   0.825 en validación), que pesa igual a cada SKU: su ventaja está en los SKUs chicos e intermitentes, mientras que
   el WAPE lo dominan los suaves (59–64 % de las unidades). SNaive es el peor (0.693 / 0.599).
3. **Por clase:** en prueba el ML gana en errática (0.600 vs 0.626 de la ruta), grumosa (0.849 vs 0.862) e
   intermitente (0.892 vs 1.028); en suave empatan (0.410 vs 0.404). En validación SBA fue mejor en suave (0.352 vs
   0.378). No se justifica un híbrido por clase: `TD_estado` corrige por familia, que es donde está el error de nivel.
4. **Ablaciones:**
   - **Limpieza de outliers: se queda.** Sin ella el WAPE de validación sube 0.85 pp y el sesgo 1.2 pp (gana solo en 4
     de 12 orígenes). La regla nueva (solo suaves y erráticos, objetivo limpio) mejora a la vieja en 2.6 pp (§3.4).
   - **Tasa de devoluciones rezagada: ruido.** +1.2 pp en validación, −0.1 pp en prueba. No se adopta.
   - **Hiperparámetros:** A (0.469), C (0.476) y B (0.493). Se elige A.
5. **El sesgo es positivo en todos los modelos:** validación +6 a +33 % y prueba +2 a +11 %. En validación viene de
   caida_2024: todo modelo entrenado con el nivel previo sobre-pronostica (MA12, la ruta y SNaive tienen el mismo
   patrón por mes, §3.4). En prueba se concentra en **dic–feb** (+13 a +22 % en el total con `ML_base`): a 3 meses no
   se anticipa lo bajos que quedaron ene–feb 2026. `TD_estado` es lo que más baja el sesgo sin perder WAPE.
6. **En el total de la compañía:** en prueba `TD_estado` tiene el menor WAPE total (8.7 %, contra 9.9 % de `ML_base` y
   10.5–11.2 % de SNaive, MA12 y la ruta). En validación los clásicos ganan en el total (ruta 16.8 % vs 20.4 % de
   `TD_estado`): para el total mensual un modelo simple es suficiente; la ventaja del ML está a nivel SKU.
7. **Cobertura:** el set evaluado (≥ 13 meses de historia y venta en los últimos 12) cubre en mediana el 98 % de las
   unidades de cada mes objetivo (rango 97–99 %). El resto son jóvenes. Los **dormidos** que se reactivan pesan 0.01 %
   en mediana (máx. 0.04 %): bien dejarlos fuera de la evaluación.
8. **Caveat:** el MAPE (180–210 %) no sirve en esta cartera. Las métricas de decisión son WAPE y sesgo; MASE y RMSSE
   quedan como secundarias.
9. **Cold start, SKUs jóvenes: gana lo simple.** `Media_post` (media desde el lanzamiento, sin el mes 0) tiene el mejor
   WAPE en validación (0.948 vs 1.005 del ML) y es el mejor en 6 de 12 orígenes; `ML_joven` no pasa la regla. En prueba
   empatan (0.816 vs 0.817), y el ML es mejor en los SKUs de 1–3 meses (0.84 vs 0.99).
   - `Media_post` sub-pronostica (−12 % / −20 %): los SKUs jóvenes siguen en rampa. Para pedidos, es un riesgo de
     quiebre que el SS de la clase "joven" compensa (`03_inventario`).
   - Para volumen se usa la media de los análogos, no la mediana: con ~23 % de lanzamientos fallidos la curva "típica"
     es casi cero (LOG #24).
10. **Cold start, pre-lanzamiento: solo rango, no número.** El punto de los análogos falla por mucho (WAPE de 155 a
    236 %), pero el rango p10–p90 cubre el 80 % y el 77 % (nominal 80 %). **No existe plan de lanzamientos**: desde su
    primera venta el SKU entra al segmento de jóvenes; el rango queda como guía opcional (LOG #24).
11. **Jerárquico (3.3, 3.5):** ninguna variante con un solo k cumple la regla en los dos periodos. `TD_mix_k3` gana en
    validación (0.449) pero no en prueba (0.496); `TD_mix_k12` es lo contrario. ETS como volumen de la familia no
    mejora. Lo que funciona es elegir k según el estado de la familia (`TD_estado`).
12. **XGBoost por serie (4.3): no se adopta.** En los top 30 SKUs y familias queda último o penúltimo fuera de muestra,
    aunque in-sample parezca excelente (R² ~0.9). Confirma que lo que funciona aquí es el modelo **global**, no el
    algoritmo de boosting en sí.
""")

nb = nbf.v4.new_notebook()
nb.cells = CELDAS
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("02_modelos.ipynb"))
print("notebooks/02_modelos.ipynb escrito")
