"""Genera notebooks/01_EDA.ipynb a partir de las celdas de abajo (fuente versionable del notebook).

Uso (desde la raíz): venv\\Scripts\\python.exe notebooks\\_construir_eda.py
Luego se ejecuta con: venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace notebooks\\01_EDA.ipynb
"""
from pathlib import Path

import nbformat as nbf

CELDAS = []
md = lambda s: CELDAS.append(nbf.v4.new_markdown_cell(s.strip()))
py = lambda s: CELDAS.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 01 · EDA — pronóstico de ventas (filial México)

Fase 1 de DECISIONES.md (§2, §12). Entrada: tablas de `data/processed/` generadas por `src/preparar.py`.

**Recordatorios de la línea base**
- Target: unidades de `Venta Producto` por `id_articulo` y mes (venta bruta, sell-in). Eje temporal = `order_date`.
- Horizonte h = 4 → en el modelado solo lags ≥ 4. Aquí, en el EDA, se mira la historia completa.
- Holdout de prueba: sep-2025 → sep-2026. Las **clasificaciones de SKU (ABC/XYZ/ADI)** de este notebook se calculan
  solo con datos **≤ ago-2025** para que puedan reutilizarse en el modelado sin leakage.
- Sin descripciones de producto ni nombres de cliente: solo claves (`id_articulo`, familia, `CLxxxx`).
""")

py("""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
from estilo import aplicar, COLORES, AZUL, GRIS, GRIS_CLARO, AZULES, MESES, FMT_MILES

aplicar()
warnings.filterwarnings("ignore", message="Mean of empty slice")   # medianas móviles sobre ventanas vacías al inicio de cada SKU
pd.set_option("display.width", 200, "display.max_columns", 40)
PROC = RAIZ / "data" / "processed"
FIG = RAIZ / "reports" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

def guardar(fig, nombre):
    fig.savefig(FIG / f"{nombre}.png")

ventas = pd.read_parquet(PROC / "ventas.parquet")
dev = pd.read_parquet(PROC / "devoluciones.parquet")
panel = pd.read_parquet(PROC / "panel.parquet")
precio = pd.read_parquet(PROC / "precio_sku_mes.parquet")
resumen = json.loads((PROC / "resumen_preparacion.json").read_text(encoding="utf-8"))

FIN_TRAIN = pd.Timestamp("2025-08-01")      # último mes antes del holdout
INICIO_TEST = pd.Timestamp("2025-09-01")
COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2022-12-01"))   # régimen externo: limpiar/interpretar, no predictor (LOG #16)
# Marcadores de reporte (no features): 2019 = pulso de destock, 2024 = escalón de destock (LOG #16)
PULSO_2019 = (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"))
ESCALON_2024 = pd.Timestamp("2024-01-01")
""")

md("## 1. Calidad de datos y limpieza")
py("""
pasos = pd.Series({k: resumen[k] for k in ["filas_crudas", "tras_dup_completos", "tras_dup_clave", "tras_corte_oct2026"]})
print(pasos.to_frame("filas").assign(quitadas=lambda d: -d.filas.diff()).to_string())
print()
print(f"Líneas Venta Producto: {resumen['venta_lineas']:,}  ·  unidades: {resumen['venta_unidades']:,.0f}")
print(f"Líneas Venta Producto con cantidad ≤ 0 (excluidas): {resumen['venta_lineas_cantidad_no_positiva']}")
print(f"Factura con fecha distinta a la orden: {resumen['pct_factura_distinta_orden']:.2%}  (desfase máx. {resumen['desfase_max_dias']} días)")
print(f"SKUs: {resumen['skus']:,}  ·  familias: {resumen['familias']}  ·  clientes (clave unificada): {resumen['clientes']:,}  ·  id_cliente crudos: {resumen['clientes_id_crudo']:,}")
print("Origen del precio SKU-mes (§9):", resumen["precio_origen"])
""")
md("""
**Lectura:** la fecha de orden y la de factura casi siempre coinciden, así que elegir `order_date` como eje (§6)
no mueve unidades entre meses de forma material.
""")

py("""
# Chequeo de la clave de cliente (src/clientes.py) contra id_cliente
m = ventas.groupby("id_cliente").cliente.nunique()
k = ventas.groupby("cliente").id_cliente.nunique()
print(f"id_cliente con >1 clave unificada: {(m > 1).sum()} de {len(m)}  ({ventas[ventas.id_cliente.isin(m[m > 1].index)].cantidad.sum() / ventas.cantidad.sum():.1%} de las unidades)")
print(f"claves unificadas que agrupan >1 id_cliente: {(k > 1).sum()} de {len(k)}")
print("id_cliente partidos en varias claves:", m[m > 1].to_dict())
""")
md("""
**Lectura:** la clave unificada junta 59 casos de un mismo cliente con varios `id_cliente`. Solo `id_cliente = 0` se
parte en muchas claves: es un id comodín que aparece en may–sep 2026 (ultimos_meses) con nombres distintos y pocas
unidades. Por eso hay más claves (3,074) que ids (2,844) sin que eso afecte el análisis.
""")

md("## 2. Serie temporal total")
py("""
mensual = ventas.groupby("mes").cantidad.sum()
anual = ventas.groupby(ventas.mes.dt.year).cantidad.sum()

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.axvspan(*COVID, color=GRIS_CLARO, alpha=.35, lw=0)
ax.axvspan(*PULSO_2019, color=COLORES[1], alpha=.08, lw=0)
ax.axvline(ESCALON_2024, color=COLORES[1], lw=1, ls=":")
ax.axvspan(INICIO_TEST, mensual.index.max(), color=AZULES[0], alpha=.6, lw=0)
ax.plot(mensual.index, mensual.values, color=AZUL)
ax.plot(mensual.index, mensual.rolling(12, center=True).mean(), color=GRIS, lw=1.5, ls="--")
ax.yaxis.set_major_formatter(FMT_MILES)
ax.set_title("Unidades vendidas por mes (Venta Producto)")
for x, t in [(COVID[0], "COVID 2020–22"), (PULSO_2019[0], "2019 pulso"), (ESCALON_2024, "2024 escalón"), (INICIO_TEST, "test")]:
    ax.text(x, ax.get_ylim()[1], t, va="top", fontsize=8, color=GRIS)
ax.set_ylabel("unidades")
guardar(fig, "01_serie_total"); plt.show()

print((anual / 1e3).round(0).to_frame("miles de unidades").assign(var_pct=(anual.pct_change() * 100).round(1)).T.to_string())
""")
md("""
Línea punteada: media móvil centrada de 12 meses. Marcadores (solo de reporte): COVID 2020–2022 (gris), pulso de destock
2019 (naranja), escalón de destock 2024 (línea punteada naranja), holdout (azul).
""")

py("""
stl = STL(np.log1p(mensual), period=12, robust=True).fit()
fig, axs = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
for ax, serie, t in zip(axs, [stl.trend, stl.seasonal, stl.resid], ["Tendencia (log)", "Estacionalidad (log)", "Residuo (log)"]):
    ax.plot(serie.index, serie.values, color=AZUL if t != "Residuo (log)" else GRIS, lw=1.5)
    ax.set_title(t, fontsize=10)
axs[2].axhline(0, color=GRIS, lw=.8)
guardar(fig, "02_stl"); plt.show()

fuerza_est = max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
fuerza_ten = max(0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())
print(f"Fuerza de estacionalidad: {fuerza_est:.2f}  ·  fuerza de tendencia: {fuerza_ten:.2f}  (0 = nada, 1 = domina)")
""")

py("""
fig, ax = plt.subplots(figsize=(9, 3.2))
plot_acf(mensual, lags=36, ax=ax, color=AZUL, vlines_kwargs={"colors": AZUL}, alpha=.05)
ax.set_title("Autocorrelación de la serie total (lags en meses)")
for l in (4, 12, 24):
    ax.axvline(l, color=GRIS_CLARO, lw=.8, ls=":")
guardar(fig, "03_acf_total"); plt.show()
print("ACF lag 4/12/24:", [round(mensual.autocorr(l), 2) for l in (4, 12, 24)])
""")

md("### 2.1 Heatmap estacional (índice mes / media del año)")
py("""
tabla = mensual.to_frame("u").assign(a=lambda d: d.index.year, m=lambda d: d.index.month)
idx = tabla.pivot(index="a", columns="m", values="u")
idx = idx.div(idx.mean(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", [COLORES[1], "#f2f1ec", AZUL])
im = ax.imshow(idx.values, cmap=cmap, norm=TwoSlopeNorm(1, .6, 1.4), aspect="auto")
ax.set_xticks(range(12), MESES); ax.set_yticks(range(len(idx)), idx.index)
for i in range(idx.shape[0]):
    for j in range(idx.shape[1]):
        if not np.isnan(idx.iat[i, j]):
            ax.text(j, i, f"{idx.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#0b0b0b")
ax.grid(False); ax.set_title("Índice estacional: unidades del mes / promedio mensual del año")
fig.colorbar(im, ax=ax, shrink=.7)
guardar(fig, "04_heatmap_estacional"); plt.show()

perfil = idx.loc[:2025].median()      # sin 2026 (año incompleto: termina en sep)
print("Índice mediano por mes (2015–2025):", dict(zip(MESES, perfil.round(2))))
t25 = tabla[tabla.a <= 2025]
h2 = t25[t25.m >= 7].groupby("a").u.sum() / t25.groupby("a").u.sum()
print(f"Participación H2 (jul–dic) por año: mediana {h2.median():.1%}, rango {h2.min():.1%}–{h2.max():.1%}")
print("Mes pico por año:", idx.idxmax(axis=1).map(lambda m: MESES[m - 1]).to_dict())
""")

md("### 2.2 Por familia")
py("""
fam_tot = ventas.groupby("familia").cantidad.sum().sort_values(ascending=False)
print(f"{len(fam_tot)} familias. Top-7 concentran {fam_tot.head(7).sum() / fam_tot.sum():.1%}; top-20 {fam_tot.head(20).sum() / fam_tot.sum():.1%} de las unidades.")
top = fam_tot.head(7).index
fm = (ventas.assign(f=ventas.familia.where(ventas.familia.isin(top), "Otros"))
      .groupby(["mes", "f"]).cantidad.sum().unstack().fillna(0)[list(top) + ["Otros"]])
anual_f = fm.groupby(fm.index.year).sum().loc[:2025]

fig, ax = plt.subplots(figsize=(11, 4))
for c, col in zip(anual_f.columns, COLORES):
    ax.plot(anual_f.index, anual_f[c], color=col if c != "Otros" else GRIS, marker="o", ms=4, label=c)
ax.yaxis.set_major_formatter(FMT_MILES); ax.legend(ncol=4, fontsize=8)
ax.set_title("Unidades anuales por familia (top 7 + Otros), 2015–2025")
guardar(fig, "05_familias_anual"); plt.show()

# ¿la forma estacional es común entre familias? correlación del perfil mensual de cada familia top con el total
def perfil_fam(s):
    t = s.to_frame("u").assign(a=s.index.year, m=s.index.month)
    t = t[t.a <= 2025].pivot(index="a", columns="m", values="u")
    return t.div(t.mean(axis=1), axis=0).median()
perfiles = pd.DataFrame({c: perfil_fam(fm[c]) for c in fm.columns})
print("Correlación del perfil estacional de cada familia con el del total:")
print(perfiles.corrwith(perfil).round(2).to_string())
""")

md("## 3. Ciclo de vida e intermitencia por SKU")
py("""
skus = panel.groupby("id_articulo").agg(familia=("familia", "first"), primera=("primera_venta", "first"),
                                         ultima=("ultima_venta", "first"))
nac = skus.primera.dt.year.value_counts().sort_index()
muer = skus[skus.ultima <= pd.Timestamp("2025-09-01")].ultima.dt.year.value_counts().sort_index()
activos = panel[panel.mes <= panel.ultima_venta].groupby("mes").id_articulo.nunique()

fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
axs[0].plot(activos.index, activos.values, color=AZUL)
axs[0].set_title("SKUs vivos por mes (entre primera y última venta)")
w = .4
axs[1].bar(nac.index - w / 2, nac.values, w, color=AZUL, label="nacen (1ª venta)")
axs[1].bar(muer.index + w / 2, muer.values, w, color=COLORES[1], label="última venta")
axs[1].set_title("Nacimientos y últimas ventas por año"); axs[1].legend(fontsize=8)
axs[1].text(.01, .97, "2015 = inicio de datos (incluye SKUs previos)", transform=axs[1].transAxes, fontsize=7, color=GRIS, va="top")
guardar(fig, "06_ciclo_vida"); plt.show()

vida_m = ((skus.ultima.dt.year - skus.primera.dt.year) * 12 + skus.ultima.dt.month - skus.primera.dt.month + 1)
print(f"Vida (meses entre 1ª y última venta): mediana {vida_m.median():.0f}, p25 {vida_m.quantile(.25):.0f}, p75 {vida_m.quantile(.75):.0f}")
print(f"SKUs obsoletos al cierre (sin venta en 12 meses a sep-2026): {resumen['skus_obsoletos_a_fin']:,} de {resumen['skus']:,}")
""")

py("""
# Métricas por SKU con datos hasta FIN_TRAIN (ago-2025), dentro de la vida activa
tr = panel[panel.mes <= FIN_TRAIN]

def metricas(g):
    y = g.unidades.to_numpy()
    nz = y[y > 0]
    n = len(y)
    adi = n / len(nz) if len(nz) else np.inf
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
    cv = y.std() / y.mean() if y.mean() > 0 else np.nan
    return pd.Series({"meses": n, "meses_con_venta": len(nz), "pct_ceros": 1 - len(nz) / n, "adi": adi,
                      "cv2": cv2, "cv": cv, "unidades": y.sum()})

met = tr.groupby("id_articulo")[["unidades"]].apply(metricas)
met = met.join(skus)

def sb(r):
    if r.meses_con_venta < 2: return "insuficiente"
    if r.adi < 1.32: return "suave" if r.cv2 < .49 else "errática"
    return "intermitente" if r.cv2 < .49 else "grumosa"
met["clase_sb"] = met.apply(sb, axis=1)
print(f"SKUs con historia ≤ ago-2025: {len(met):,}")
print(met.clase_sb.value_counts().to_frame("skus").assign(pct=lambda d: (d.skus / d.skus.sum() * 100).round(1),
      pct_unidades=met.groupby("clase_sb").unidades.sum() / met.unidades.sum() * 100).round(1).to_string())
""")

py("""
vis = met[met.clase_sb != "insuficiente"]
fig, ax = plt.subplots(figsize=(7, 5))
for c, col in zip(["suave", "errática", "intermitente", "grumosa"], COLORES):
    d = vis[vis.clase_sb == c]
    ax.scatter(d.adi, d.cv2, s=10, color=col, alpha=.55, label=f"{c} ({len(d)})", lw=0)
ax.axvline(1.32, color=GRIS, lw=1, ls="--"); ax.axhline(.49, color=GRIS, lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=.1); ax.set_ylim(bottom=0)
ax.set_xlabel("ADI (meses entre ventas)"); ax.set_ylabel("CV² del tamaño de la demanda")
ax.set_title("Clasificación Syntetos–Boylan (datos ≤ ago-2025)"); ax.legend(fontsize=8)
guardar(fig, "07_adi_cv2"); plt.show()

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(met.pct_ceros, bins=20, color=AZUL, edgecolor="#fcfcfb")
ax.set_title("% de meses en cero por SKU (dentro de vida activa)"); ax.set_xlabel("% meses en cero")
guardar(fig, "08_pct_ceros"); plt.show()
""")

md("""
### 3.1 ABC × XYZ

- **ABC** por valor de consumo (unidades × precio MXN) en los **últimos 12 meses de train** (sep-2024 → ago-2025),
  para SKUs vivos en ese periodo. A = 80 % acumulado del valor, B = siguiente 15 %, C = último 5 %.
- **XYZ** por CV de la demanda mensual en los últimos 24 meses de train: X < 0.5, Y 0.5–1.0, Z > 1.0.
""")
py("""
ini12, ini24 = FIN_TRAIN - pd.DateOffset(months=11), FIN_TRAIN - pd.DateOffset(months=23)
v12 = ventas[(ventas.mes >= ini12) & (ventas.mes <= FIN_TRAIN)].groupby(["id_articulo", "mes"]).cantidad.sum().reset_index()
v12 = v12.merge(precio, on=["id_articulo", "mes"], how="left")
valor = (v12.cantidad * v12.precio_mxn).groupby(v12.id_articulo).sum().sort_values(ascending=False)
acum = valor.cumsum() / valor.sum()
abc = pd.cut(acum, [0, .8, .95, 1.0001], labels=["A", "B", "C"]).rename("abc")

p24 = panel[(panel.mes >= ini24) & (panel.mes <= FIN_TRAIN) & panel.id_articulo.isin(abc.index)]
cv24 = p24.groupby("id_articulo").unidades.agg(lambda s: s.std() / s.mean() if s.mean() > 0 else np.nan)
xyz = pd.cut(cv24, [-np.inf, .5, 1.0, np.inf], labels=["X", "Y", "Z"]).rename("xyz")

clas = pd.concat([abc, xyz, valor.rename("valor_mxn_12m")], axis=1).join(met[["clase_sb", "adi", "cv2", "pct_ceros", "familia"]])
n = pd.crosstab(clas.abc, clas.xyz, margins=True, margins_name="total")
val = pd.crosstab(clas.abc, clas.xyz, values=clas.valor_mxn_12m, aggfunc="sum", normalize=True).mul(100).round(1)
print("SKUs por celda:"); print(n.to_string()); print()
print("% del valor de consumo por celda:"); print(val.to_string()); print()
print("Clase Syntetos–Boylan dentro de cada ABC (n SKUs):")
print(pd.crosstab(clas.abc, clas.clase_sb).to_string())
clas.reset_index().to_parquet(PROC / "clasificacion_sku.parquet", index=False)
""")

md("""
## 4. Anomalías y regímenes

Marco (LOG #16): 2019 = **pulso** de destock, 2024 = **escalón** de destock; 2025 no es una contracción aparte.
COVID 2020–2022 es régimen externo (sirve para limpiar e interpretar, no como predictor). Ninguno de estos marcadores
entra al modelo como flag. La parte predictiva va por el **proxy de sobre-stock** (§4.6).
""")
py("""
ym = mensual.to_frame("u").assign(a=mensual.index.year, m=mensual.index.month)
piv = ym.pivot(index="m", columns="a", values="u")
yoy = (piv / piv.shift(axis=1) - 1) * 100
print("Variación % vs mismo mes del año anterior (años de interés):")
print(yoy[[2019, 2020, 2021, 2022, 2024, 2025, 2026]].round(1).set_axis(MESES).T.to_string())

# ¿la contracción viene de menos SKUs vendiendo o de menos unidades por SKU?
por_a = ventas[ventas.mes.dt.year <= 2025].groupby(ventas.mes.dt.year).agg(u=("cantidad", "sum"), skus=("id_articulo", "nunique"),
                                                                          clientes=("cliente", "nunique"), lineas=("cantidad", "size"))
por_a["u_por_sku"] = por_a.u / por_a.skus
por_a["u_por_cliente"] = por_a.u / por_a.clientes
print(); print(por_a.round(0).astype(int).to_string())
""")

md("""
### 4.1 Oct-2018: ¿atípico o compra anticipada?

Exceso de oct-2018 = unidades de oct-2018 − mediana de los octubres vecinos (2016, 2017, 2019), desglosado por
cliente, familia y SKU. Después se revisa si quienes generaron el exceso compraron menos en los 12 meses siguientes
(destock de 2019). **No se borra el dato:** se trata como señal.
""")
py('''
def exceso(mes, refs, por):
    """Unidades del mes − mediana del mismo mes en los años de referencia, por la dimensión `por`."""
    t = ventas.groupby([por, "mes"]).cantidad.sum().unstack(fill_value=0)
    meses_ref = [pd.Timestamp(f"{a}-{mes.month:02d}-01") for a in refs]
    return (t[mes] - t[meses_ref].median(axis=1)).sort_values(ascending=False)

M18 = pd.Timestamp("2018-10-01")
exc_total = mensual[M18] - np.median([mensual[pd.Timestamp(f"{a}-10-01")] for a in (2016, 2017, 2019)])
print(f"Oct-2018: {mensual[M18]:,.0f} u.  Exceso vs octubres vecinos: {exc_total:,.0f} u.")
for por in ["cliente", "familia", "id_articulo"]:
    e = exceso(M18, (2016, 2017, 2019), por)
    print(f"  top-3 {por}: {e.head(3).round(0).astype(int).to_dict()}  → {e.head(3).sum() / exc_total:.0%} del exceso; "
          f"top-10: {e.head(10).sum() / exc_total:.0%}")

e_cli = exceso(M18, (2016, 2017, 2019), "cliente")
top_cli = e_cli.head(3).index
mc = ventas[ventas.cliente.isin(top_cli)].groupby(["mes", "cliente"]).cantidad.sum().unstack(fill_value=0)
antes, despues = mc.loc["2017-10-01":"2018-09-01"].sum(), mc.loc["2018-11-01":"2019-10-01"].sum()
print("\\nClientes del exceso: 12 m antes (oct17–sep18) vs oct-2018 vs 12 m después (nov18–oct19):")
print(pd.DataFrame({"12m_antes": antes, "oct18": mc.loc[M18], "12m_despues": despues,
                    "var_%": (despues / antes - 1) * 100}).round(0).astype(int).to_string())

e_sku = exceso(M18, (2016, 2017, 2019), "id_articulo").head(20).index
ms = ventas[ventas.id_articulo.isin(e_sku)].groupby("mes").cantidad.sum()
print(f"\\nTop-20 SKUs del exceso: 12 m antes {ms.loc['2017-10-01':'2018-09-01'].sum():,.0f} · oct-18 {ms.get(M18, 0):,.0f} · "
      f"12 m después {ms.loc['2018-11-01':'2019-10-01'].sum():,.0f}")

fig, ax = plt.subplots(figsize=(11, 3.6))
z = mc.loc["2017-01-01":"2020-06-01"]
for c, col in zip(top_cli, COLORES):
    ax.plot(z.index, z[c], color=col, label=c)
ax.axvline(M18, color=GRIS, lw=1, ls=":"); ax.yaxis.set_major_formatter(FMT_MILES); ax.legend(fontsize=8)
ax.set_title("Clientes que generan el exceso de oct-2018: unidades por mes")
guardar(fig, "12_oct2018_clientes"); plt.show()
''')

md("### 4.2 Escalón 2024: ¿sobre-compra previa en 2023?")
py('''
nivel = mensual.groupby(mensual.index.year).mean()
print("Promedio mensual 2021–2023: {:,.0f}  ·  2024–2025: {:,.0f}  ({:+.1%})".format(
    nivel.loc[2021:2023].mean(), nivel.loc[2024:2025].mean(), nivel.loc[2024:2025].mean() / nivel.loc[2021:2023].mean() - 1))
M23 = pd.Timestamp("2023-10-01")
exc23 = mensual[M23] - np.median([mensual[pd.Timestamp(f"{a}-10-01")] for a in (2021, 2022)])
e23 = exceso(M23, (2021, 2022), "cliente")
print(f"Oct-2023: {mensual[M23]:,.0f} u., exceso vs oct-2021/22: {exc23:,.0f} u.; top-3 clientes: {e23.head(3).round(0).astype(int).to_dict()}")
print("(El detalle por cliente de la caída de 2024 está en §5.)")
''')

md("""
### 4.3 COVID 2020–2022 por familia

Demanda anual por familia, indexada a 2019 = 100. Se muestran las familias con ≥ 0.5 % de las unidades de 2019–2023.
El efecto no es uniforme: hay familias que suben (home-office) y otras que bajan (enterprise).
""")
py('''
fa = ventas[ventas.mes.dt.year.between(2019, 2023)].groupby([ventas.mes.dt.year.rename("a"), "familia"]).cantidad.sum().unstack(0).fillna(0)
rel = fa[fa.sum(axis=1) / fa.values.sum() >= .005]
ind = rel.div(rel[2019], axis=0).mul(100).replace(np.inf, np.nan)
ind["covid_vs_2019_%"] = (rel[[2020, 2021, 2022]].mean(axis=1) / rel[2019] - 1) * 100
ind["share_19_23_%"] = rel.sum(axis=1) / fa.values.sum() * 100
ind = ind.sort_values("covid_vs_2019_%", ascending=False)
print(f"{len(rel)} familias = {rel.values.sum() / fa.values.sum():.0%} de las unidades 2019–2023")
print(ind.round(0).to_string())

orden = ind["covid_vs_2019_%"].dropna().clip(-100, 300)
fig, ax = plt.subplots(figsize=(8, max(3, .28 * len(orden))))
ax.barh(orden.index[::-1], orden.values[::-1], color=[AZUL if v >= 0 else COLORES[1] for v in orden.values[::-1]], height=.7)
ax.axvline(0, color=GRIS, lw=.8); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("promedio 2020–2022 vs 2019 (%), recortado a 300")
ax.set_title("Efecto COVID por familia (azul sube, naranja baja)")
guardar(fig, "13_covid_familias"); plt.show()

top8 = rel[2019].sort_values(ascending=False).head(8).index
fig, ax = plt.subplots(figsize=(10, 3.8))
for f, col in zip(top8, COLORES):
    ax.plot(ind.columns[:5], ind.loc[f, ind.columns[:5]], color=col, marker="o", ms=4, label=f)
ax.axhline(100, color=GRIS, lw=.8); ax.set_ylabel("índice (2019 = 100)"); ax.legend(ncol=4, fontsize=8)
ax.set_title("Demanda anual de las 8 familias más grandes, 2019 = 100")
guardar(fig, "14_covid_top8"); plt.show()
''')

md("""
**Perfil de familia** (`data/perfil_familia.csv`, curado a mano a partir de `descripcion1` del catálogo; aquí solo
se usan las claves). Perfiles: *enterprise* (cableado estructurado, racks, switches, PoE, KVM, tarjetas
seriales, candados, video corporativo); *home_office* (cables y adaptadores de video, soportes de TV/monitor,
audífonos, cámaras, teclado/mouse, hubs, docks, adaptadores de red USB); *mixto* (PC/consumo general).
La familia se conserva: el perfil es una agrupación adicional para interpretar COVID, no un feature.
""")
py('''
perfil = pd.read_csv(RAIZ / "data" / "perfil_familia.csv", dtype=str).set_index("familia")
ventas["perfil"] = ventas.familia.map(perfil.perfil).fillna("mixto")
pa = ventas[ventas.mes.dt.year <= 2025].groupby([ventas.mes.dt.year.rename("a"), "perfil"]).cantidad.sum().unstack().drop(columns="no_producto", errors="ignore")
ip = pa.div(pa.loc[2019]).mul(100)
print("Participación de unidades 2019:", (pa.loc[2019] / pa.loc[2019].sum() * 100).round(1).to_dict())
print("Índice anual por perfil (2019 = 100):"); print(ip.round(0).astype(int).T.to_string())

fig, ax = plt.subplots(figsize=(10, 3.8))
for p_, col in zip(["enterprise", "home_office", "mixto"], [COLORES[1], AZUL, GRIS]):
    ax.plot(ip.index, ip[p_], color=col, marker="o", ms=4, label=p_)
ax.axvspan(2020, 2022, color=GRIS_CLARO, alpha=.35, lw=0); ax.axhline(100, color=GRIS, lw=.8)
ax.set_ylabel("índice (2019 = 100)"); ax.legend(fontsize=8)
ax.set_title("Demanda anual por perfil de familia (banda: COVID 2020–2022)")
guardar(fig, "14b_covid_perfil"); plt.show()

ind["perfil"] = ind.index.map(perfil.perfil)
print("\\nFamilias relevantes por perfil y dirección del efecto COVID (n):")
print(pd.crosstab(ind.perfil, np.where(ind["covid_vs_2019_%"] >= 0, "sube", "baja")).to_string())
''')

md("""
### 4.4 Estacionalidad robusta

Índice = unidades del mes / mediana móvil centrada de 13 meses (quita nivel y tendencia), resumido por mes del año
con la **mediana** entre años (robusta a oct-2018 o abr-2020) frente a la **media** (clásica). Se excluyen
COVID (2020–2022) y el holdout, y se calcula solo con train (≤ ago-2025). Los índices por familia se guardan
para usarlos en el feature engineering.
""")
py('''
def indice_robusto(s, excluir=COVID):
    s = s[s.index <= FIN_TRAIN]
    r = (s / s.rolling(13, center=True, min_periods=7).median()).replace([np.inf, -np.inf], np.nan)
    r = r[(r.index < excluir[0]) | (r.index > excluir[1])].dropna()
    med = r.groupby(r.index.month).median()
    mea = r.groupby(r.index.month).mean()
    return med / med.mean(), mea / mea.mean(), r.groupby(r.index.month).size()

med, mea, n_anos = indice_robusto(mensual)
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(range(12), mea.values, color=GRIS, marker="o", ms=4, label="media (clásico)")
ax.plot(range(12), med.values, color=AZUL, marker="o", ms=5, label="mediana (robusto)")
ax.axhline(1, color=GRIS_CLARO, lw=.8); ax.set_xticks(range(12), MESES); ax.legend(fontsize=8)
ax.set_title("Índice estacional total: clásico vs robusto (train, sin COVID)")
guardar(fig, "15_estacional_robusto"); plt.show()
print(pd.DataFrame({"robusto": med, "clasico": mea, "dif": med - mea}).set_axis(MESES).round(2).T.to_string())

fam_m = ventas.groupby(["familia", "mes"]).cantidad.sum().unstack(0).reindex(mensual.index, fill_value=0).fillna(0)
filas = []
for f in fam_m.columns:
    s = fam_m[f]
    if (s[s.index <= FIN_TRAIN] > 0).sum() < 36:        # mínimo 3 años con venta para un índice confiable
        continue
    i, _, n = indice_robusto(s)
    if len(i) == 12:
        filas.append(pd.DataFrame({"familia": f, "mes_del_ano": i.index, "indice": i.values, "n_obs": n.values}))
ie = pd.concat(filas, ignore_index=True)
ie.to_parquet(PROC / "indice_estacional_familia.parquet", index=False)
print(f"Índice robusto guardado para {ie.familia.nunique()} familias (≥ 36 meses con venta en train).")
''')

md("""
### 4.5 Outliers por SKU

Para cada SKU, en log(1 + unidades): residuo frente a la mediana móvil centrada de 13 meses, escalado por la MAD
del SKU (con piso de 0.25 para series con muchos ceros). Outlier = z > 3.5 con venta > 0. **Se marca, no se borra.**
La ventana centrada sirve para limpiar/interpretar la historia; en el backtest se recalcula solo con train.
""")
py('''
p = panel[["id_articulo", "familia", "mes", "unidades"]].copy()
p["l"] = np.log1p(p.unidades)
p["base"] = p.groupby("id_articulo").l.transform(lambda s: s.rolling(13, center=True, min_periods=7).median())
p["res"] = p.l - p.base
esc = p.groupby("id_articulo").res.transform(lambda r: max(1.4826 * (r - r.median()).abs().median(), .25))
p["z"] = p.res / esc
p["outlier"] = (p.z > 3.5) & (p.unidades > 0)
p[["id_articulo", "mes", "unidades", "z", "outlier"]].to_parquet(PROC / "outliers_sku.parquet", index=False)

o = p[p.outlier]
print(f"SKU-mes marcados: {len(o):,} de {(p.unidades > 0).sum():,} con venta ({len(o) / (p.unidades > 0).sum():.1%}); "
      f"{o.unidades.sum() / p.unidades.sum():.1%} de las unidades; {o.id_articulo.nunique():,} SKUs")
um = o.groupby("mes").unidades.sum().reindex(mensual.index, fill_value=0) / mensual
print("Meses con más unidades en outliers (% del mes):", (um.sort_values(ascending=False).head(6) * 100).round(1).to_dict())

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(um.index, um.values * 100, color=AZUL, lw=1.5)
ax.axvspan(*COVID, color=GRIS_CLARO, alpha=.35, lw=0)
ax.set_ylabel("% de unidades del mes"); ax.set_title("Peso de los SKU-mes outlier en cada mes")
guardar(fig, "16_outliers_mes"); plt.show()
''')

md("""
### 4.6 Proxy de sobre-stock (sobre-disparo)

Idea: un cliente que compra por encima de su ritmo "normal" acumula inventario y después compra menos (destock).
- **Base local adaptativa** b_t = mediana de los 12 meses previos (t−12 … t−1); si es 0, la media (SKUs intermitentes).
- **Exceso acumulado** E_t = Σ_{k=0..5}(y_{t−k} − b_{t−k}) / b_t, en "meses de demanda base" comprados de más.
- **Chequeo predictivo sin leakage:** origen T = último mes conocido, objetivo T+4 (h = 4). Se compara E_T con
  log(y_{T+4} / b_T). La simple reversión a la media daría ≈ 0; un destock real da **negativo** cuando E_T es alto.
  La validación formal va en el backtest (fase 3).
""")
py('''
def proxy(y):
    b = y.shift(1).rolling(12, min_periods=6).median()
    b = b.where(b > 0, y.shift(1).rolling(12, min_periods=6).mean())
    E = (y - b).rolling(6, min_periods=6).sum() / b
    objetivo = np.log1p(y.shift(-4)) - np.log1p(b)
    return pd.DataFrame({"y": y, "b": b, "E": E, "objetivo": objetivo})

ag = proxy(mensual)
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(ag.index, ag.E, color=AZUL)
ax.axhline(0, color=GRIS, lw=.8); ax.axvspan(*PULSO_2019, color=COLORES[1], alpha=.08, lw=0)
ax.axvline(ESCALON_2024, color=COLORES[1], lw=1, ls=":"); ax.axvspan(*COVID, color=GRIS_CLARO, alpha=.35, lw=0)
ax.set_ylabel("meses de demanda base"); ax.set_title("Exceso acumulado de 6 meses vs base local (total)")
guardar(fig, "17_proxy_total"); plt.show()
v = ag.dropna()
v_tr = v[v.index <= FIN_TRAIN - pd.DateOffset(months=4)]
print(f"Total: corr. Spearman E_T vs log(y_T+4 / b_T) = {v_tr.E.corr(v_tr.objetivo, method='spearman'):.2f} (n = {len(v_tr)}, train)")

# SKU y familia (agregado). Solo train y bases con volumen mínimo para evitar ruido de cola.
def panel_proxy(df, clave, b_min):
    out = []
    for k, g in df.groupby(clave):
        s = g.set_index("mes").unidades.asfreq("MS", fill_value=0)
        r = proxy(s).assign(**{clave: k})
        out.append(r)
    r = pd.concat(out).dropna()
    return r[(r.index <= FIN_TRAIN - pd.DateOffset(months=4)) & (r.b >= b_min)]

ps = panel_proxy(panel, "id_articulo", 5)
pf = panel_proxy(fam_m.stack().rename("unidades").reset_index(), "familia", 50)
for nombre, d in [("SKU", ps), ("familia", pf)]:
    q = pd.qcut(d.E.rank(method="first"), 5, labels=["Q1 bajo", "Q2", "Q3", "Q4", "Q5 alto"])
    print(f"{nombre}: corr. Spearman = {d.E.corr(d.objetivo, method='spearman'):.2f} (n = {len(d):,}); "
          f"mediana del objetivo por quintil de E: {d.groupby(q, observed=True).objetivo.median().round(2).to_dict()}")

q = pd.qcut(ps.E.rank(method="first"), 5, labels=["Q1 bajo", "Q2", "Q3", "Q4", "Q5 alto"])
m = ps.groupby(q, observed=True).objetivo.median()
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(m.index.astype(str), m.values, color=AZUL, width=.6)
ax.axhline(0, color=GRIS, lw=.8)
ax.set_ylabel("mediana log(y_T+4 / b_T)"); ax.set_xlabel("quintil del exceso acumulado E_T (SKU)")
ax.set_title("Demanda a h = 4 vs base, según el sobre-disparo previo (SKU, train)")
guardar(fig, "18_proxy_sku_quintiles"); plt.show()
''')

md("## 5. Puente de unidades por cliente (año contra año)")
py("""
cy = ventas[ventas.mes.dt.year <= 2025].groupby([ventas.mes.dt.year.rename("a"), "cliente"]).cantidad.sum().unstack(0).fillna(0)

def puente(a):
    prev, cur = cy[a - 1], cy[a]
    d = cur - prev
    return pd.Series({"año_previo": prev.sum(),
                      "nuevos": cur[(prev == 0) & (cur > 0)].sum(),
                      "expansión": d[(prev > 0) & (cur > 0) & (d > 0)].sum(),
                      "contracción": d[(prev > 0) & (cur > 0) & (d < 0)].sum(),
                      "perdidos": -prev[(prev > 0) & (cur == 0)].sum(),
                      "año": cur.sum(),
                      "n_nuevos": int(((prev == 0) & (cur > 0)).sum()), "n_perdidos": int(((prev > 0) & (cur == 0)).sum())})

pt = pd.DataFrame({a: puente(a) for a in range(2016, 2026)}).T
print((pt[["año_previo", "nuevos", "expansión", "contracción", "perdidos", "año"]] / 1e3).round(0).astype(int).to_string())
print(pt[["n_nuevos", "n_perdidos"]].astype(int).T.to_string())

comp = ["nuevos", "expansión", "contracción", "perdidos"]
fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(pt)); w = .2
for i, (c, col) in enumerate(zip(comp, [COLORES[2], AZUL, COLORES[1], COLORES[7]])):
    ax.bar(x + (i - 1.5) * w, pt[c] / 1e3, w, color=col, label=c)
ax.plot(x, (pt.año - pt.año_previo) / 1e3, color="#0b0b0b", marker="o", ms=5, lw=1.5, label="cambio neto")
ax.axhline(0, color=GRIS, lw=.8); ax.set_xticks(x, pt.index); ax.set_ylabel("miles de unidades")
ax.set_title("Puente de unidades por cliente vs año anterior"); ax.legend(ncol=5, fontsize=8)
guardar(fig, "09_puente_clientes"); plt.show()
""")

py("""
# 2019 (pulso), 2024 (escalón) y 2025: ¿qué clientes explican la caída?
for a in (2019, 2024, 2025):
    d = (cy[a] - cy[a - 1]).sort_values()
    caida = d[d < 0]
    print(f"\\n{a}: cambio neto {d.sum() / 1e3:,.0f} k unidades. Top-5 clientes que más caen ({caida.head(5).sum() / caida.sum():.0%} de la caída bruta):")
    print(pd.DataFrame({"prev": cy[a - 1], "año": cy[a], "delta": d}).loc[caida.head(5).index].round(0).astype(int).to_string())

conc = cy.loc[:, 2025].sort_values(ascending=False)
print(f"\\nConcentración 2025: top-5 clientes {conc.head(5).sum() / conc.sum():.1%}, top-20 {conc.head(20).sum() / conc.sum():.1%}")
hhi = (cy / cy.sum()).pow(2).sum() * 1e4
print("HHI por año (10,000 = un solo cliente):", hhi.round(0).astype(int).to_dict())
""")

md("## 6. Devoluciones (sub-análisis, §10)")
py("""
dev["a"] = dev.mes.dt.year
da = dev[dev.a <= 2025].pivot_table(index="a", columns="concepto", values="unidades_dev", aggfunc="sum").fillna(0)
da["total"] = da.sum(axis=1)
tasa = (da.div(anual.loc[:2025], axis=0) * 100).round(2)
print("Unidades devueltas por año y concepto:"); print(da.round(0).astype(int).to_string())
print("\\nTasa = unidades devueltas / unidades vendidas del mismo año (%), neteo anual:"); print(tasa.to_string())

fig, ax = plt.subplots(figsize=(10, 3.6))
cols = [c for c in da.columns if c != "total"]
for c, col in zip(cols, COLORES):
    ax.plot(tasa.index, tasa[c], color=col, marker="o", ms=4, label=c)
ax.set_ylabel("% de unidades vendidas"); ax.legend(fontsize=8, ncol=2)
ax.set_title("Tasa anual de devolución por concepto")
guardar(fig, "10_tasa_devolucion"); plt.show()
""")

py("""
real = dev[dev.concepto.isin(["Devolución de producto", "Garantía de Producto", "Cargos por devolucion (vtas)"])]
# estacionalidad de devoluciones: índice mensual
dm = real.groupby("mes").unidades_dev.sum().reindex(mensual.index, fill_value=0)
t = dm.to_frame("u").assign(a=dm.index.year, m=dm.index.month)
t = t[t.a <= 2025].pivot(index="a", columns="m", values="u")
print("Índice estacional mediano de devoluciones (sin rotación):", dict(zip(MESES, t.div(t.mean(axis=1), axis=0).median().round(2))))

# concentración por cliente
c = real.groupby("cliente").unidades_dev.sum().sort_values(ascending=False)
print(f"Top-5 clientes concentran {c.head(5).sum() / c.sum():.1%} de las devoluciones (sin rotación); top-20 {c.head(20).sum() / c.sum():.1%}")
tc = (c / ventas.groupby("cliente").cantidad.sum()).dropna()
print("Tasa de devolución de los 5 clientes que más devuelven:", (tc.loc[c.head(5).index] * 100).round(1).to_dict())

# desfase venta → devolución: última venta del mismo SKU al mismo cliente antes de la devolución
s = ventas[["order_date", "cliente", "id_articulo"]].sort_values("order_date")
r = real[["order_date", "cliente", "id_articulo"]].sort_values("order_date")
emp = pd.merge_asof(r, s.rename(columns={"order_date": "fecha_venta"}), left_on="order_date", right_on="fecha_venta",
                    by=["cliente", "id_articulo"], direction="backward")
lag = (emp.order_date - emp.fecha_venta).dt.days / 30.44
print(f"Devoluciones con venta previa del mismo SKU-cliente: {lag.notna().mean():.1%}")
print("Desfase (meses) venta→devolución:", lag.describe(percentiles=[.25, .5, .75, .9]).round(1)[["25%", "50%", "75%", "90%"]].to_dict())

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(lag.dropna().clip(upper=24), bins=48, color=AZUL, edgecolor="#fcfcfb")
ax.set_xlabel("meses desde la última venta del mismo SKU al mismo cliente (cortado en 24)")
ax.set_title("Desfase venta → devolución")
guardar(fig, "11_desfase_devolucion"); plt.show()
""")

md("## 7. Hallazgos")
md("""
**Datos y limpieza**
- 1.40 M filas → 1.31 M líneas de Venta Producto (18.2 M unidades, ene-2015 → sep-2026). Se quitaron 1,209 duplicados
  y oct-2026 (parcial). Orden y factura difieren solo en 0.16 % de las líneas → el eje `order_date` es seguro.
- Precio MXN disponible en > 99.9 % de los SKU-mes; la cascada de §9 casi no se usa.

**Nivel y tendencia**
- Sin crecimiento estructural: 1.4–1.8 M unidades/año. Marcadores de reporte (LOG #16): **2019 = pulso** de destock
  (−18.5 %) y **2024 = escalón** de destock (−15.9 %; el promedio mensual pasa de 139 k en 2021–23 a 118 k en 2024–25).
  2025 (−2.2 %) es el mismo nivel bajo, no una contracción aparte.
- COVID 2020–2022 (régimen externo): abr–may 2020 caen −49 % / −58 % y 2021 rebota (+20 %).

**Anomalías (§4.1–4.5)**
- **Oct-2018 (278 k u.) es compra anticipada, no ruido.** El exceso frente a los octubres vecinos es de 120 k u.:
  CL0001 + CL0004 + CL0008 generan el 78 %, la familia A05 el 56 % y un solo SKU (1597) el 53 %. Esos clientes
  compraron 17 % / 4 % / 15 % menos en los 12 meses siguientes. Es el arranque del pulso de 2019 (CL0001 −189 k y
  CL0004 −105 k en 2019). Se conserva como señal y se marca como outlier para que no infle los índices.
- **2024 repite el patrón:** oct-2023 (203 k u.) trae un exceso de 48 k, liderado por CL0003, que en 2024 es el
  cliente que más cae (−112 k). Además, CL0004 cae −63 k y CL0017 se pierde (−49 k).
- **Estacionalidad robusta:** con medianas y sin COVID, el índice de octubre baja de 1.37 (clásico) a **1.22** y el de
  diciembre de 0.97 a 0.89. El índice clásico está inflado por oct-2018. Se guardan índices robustos para 44 familias
  (`indice_estacional_familia.parquet`).
- **Outliers por SKU:** 6.0 % de los SKU-mes con venta (5.7 % de las unidades). Los meses donde más pesan son
  oct-2018 (36 % del mes), nov-2022 (32 %), dic-2018, ago-2023 y nov-2023: coinciden con los excesos previos a
  cada destock. Se guardan en `outliers_sku.parquet` (marcados, no borrados).

**COVID por familia (§4.3), 2020–2022 vs 2019**
- Suben: A08 (+181 %), A06 (+67 %), E01, C12, A05, C02, C03 (+28–45 %). E01, C12 y C02 regresan a su nivel o quedan
  por debajo en 2023 (joroba temporal). A06, A05 y A08 siguen arriba (cambio estructural).
- Bajan: A09, D03, C04, B01, B04 (−10 a −19 %). B04 es la familia más grande (17 %). B02 cae en 2020 y se
  recupera después.
- **Por perfil** (`data/perfil_familia.csv`): de las 23 familias relevantes, las 9 *home_office* suben y 8 de 9
  *enterprise* bajan (la excepción es B05, +7 %). Índice 2019 = 100:
  - *enterprise* (46 % de las unidades de 2019): 80 en 2020, se recupera a 96–104 en 2021–23 y cae a 79–83 en
    2024–25. El escalón de 2024 es sobre todo enterprise.
  - *home_office* (25 %): 104 → 151 → 155 → 163, y sigue en ~146 en 2024–25. **El salto de COVID no se revirtió**,
    es un cambio de nivel estructural. (Su 145 de 2018 es el pico de oct-2018, SKU 1597.)
  - *mixto* (29 %): tendencia descendente de largo plazo (149 en 2015 → 79 en 2025), sin efecto COVID claro.
  → Para limpiar 2020–2022 no basta un ajuste uniforme: la dirección depende del perfil.

**Proxy de sobre-stock (§4.6), chequeo preliminar en train**
- **Total:** la correlación es negativa (Spearman −0.20). Tras un sobre-disparo, la demanda a h = 4 queda bajo la base.
  El gráfico muestra picos de exceso antes de 2019 y antes de 2024. Es la firma del destock.
- **SKU y familia:** la correlación es **positiva** (+0.21 / +0.22) porque domina el ciclo de vida. SKUs que suben
  siguen subiendo y SKUs que se apagan siguen cayendo (Q1 de E → −0.62).
  → La versión cruda no sirve tal cual a nivel SKU. Para la fase de features: medir el exceso contra una base con
  tendencia (Holt/EWMA adaptativo), separar pulso corto (3 m) de tendencia y usar el proxy **agregado** como señal
  común para los SKUs. Todas las variantes se validan en el backtest walk-forward.

**Estacionalidad**
- Moderada (fuerza STL 0.42). Patrón mediano: ago–oct fuerte (pico en **oct, índice 1.19**), abril y enero bajos
  (0.81 / 0.88). H2 = 54 % del año (rango 47–60 %). El mes pico cambia de un año a otro.
- Los perfiles de las familias top se parecen al del total (correlación 0.62–0.90). Eso respalda usar un índice
  estacional de familia/total para SKUs con poca historia.
- ACF del total: lag 4 ≈ 0 y lag 12 ≈ 0.30. Con h = 4 el lag-4 aporta poco por sí solo; pesan más el nivel
  (ventanas desfasadas) y el lag-12.

**Intermitencia y segmentación (datos ≤ ago-2025)**
- 2,339 SKUs. **Suaves: 7 % de los SKUs pero 63 % de las unidades**; erráticos 25 % / 32 %. Intermitentes y
  grumosos son el 56 % de los SKUs pero solo el 5 % de las unidades.
  → El volumen se gana con ETS/SARIMA/modelo global sobre suaves y erráticos. Croston/SBA/TSB aplica a la cola larga.
- ABC×XYZ: 208 SKUs A = 80 % del valor; **AX (71 SKUs) = 41 % del valor**. CZ = 330 SKUs con < 3 % del valor.
- Vida mediana de un SKU: 43 meses. 1,532 de 2,417 SKUs están obsoletos a sep-2026. Hay unos 880–1,150 SKUs
  vendiendo cada año. Hay rotación de catálogo alta, así que importan el arranque de SKUs nuevos y el decaimiento
  (TSB).

**Clientes (diagnóstico)**
- Muy concentrado: en 2025 el top-5 = 76 % y el top-20 = 98 % de las unidades.
- 2019: CL0001 (−189 k) y CL0004 (−105 k) explican la mayor parte de la caída. 2024: contracción de clientes existentes
  (−253 k) y clientes perdidos (−61 k). 2025: CL0004 casi desaparece (41 k → 6 k) y quedan 276 clientes activos
  (el mínimo de la serie).
  → Un cambio en 2–3 mayoristas mueve el total. Esto respalda la regla de **ajuste manual** de §8.

**Devoluciones (§10)**
- Tasa anual total (neteo anual) de 2.0–5.4 %, **creciente desde 2022** (2.35 % → 5.40 % en 2025). Suben tanto
  Devolución de producto (0.56 % → 1.99 %) como Rotación de inventarios (≈ 3.4–3.9 % en 2024–25).
- Concentradas: el top-5 de clientes = 70 % de las devoluciones (sin rotación). CL0022 devolvió el 92 % de lo que compró.
- Desfase: en 99 % de los casos hay una venta previa del mismo SKU al mismo cliente. La mediana es de ~0.1 meses
  desde la venta más reciente y el p90 de 2.4 meses (medido contra la venta *más reciente*, así que es un límite
  inferior). Las devoluciones caen cerca de la venta: una **tasa rezagada ≥ 4** como feature es viable para el
  experimento de baseline.
- Estacionalidad de devoluciones: pico en enero (1.20), más bajas en abr / jul / dic.

**Salida para la fase 2:** `data/processed/clasificacion_sku.parquet` (ABC, XYZ, clase Syntetos–Boylan, ADI, CV² por
SKU, calculados solo con train).
""")

nb = nbf.v4.new_notebook()
nb.cells = CELDAS
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("01_EDA.ipynb"))
print("notebooks/01_EDA.ipynb escrito")
