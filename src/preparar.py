"""Preparación: de data/full_2015_2026.parquet (salida de merge_csvs.py) a las tablas limpias de data/processed/.

Aplica DECISIONES.md §3–§10 y LOG_DECISIONES.md (2026-10-22):
- deduplicado (§5), corte de oct-2026 (mes parcial), eje temporal = order_date (§6);
- se eliminan descripciones, nombres de cliente y región (§4); se conservan claves (id_articulo, familia, cliente);
- target = concepto 'Venta Producto' con cantidad > 0 (§3); devoluciones a su propia tabla (§10);
- precio unitario MXN con imputación en cascada, solo para ABC (§9);
- panel SKU × mes con ceros dentro de la vida activa (§7).

Salidas (data/processed/):
    ventas.parquet         líneas de Venta Producto limpias
    devoluciones.parquet   líneas de devolución / garantía / rotación (unidades en positivo)
    precio_sku_mes.parquet precio unitario MXN por SKU y mes (con su origen de imputación)
    panel.parquet          SKU × mes: unidades, flags de vida activa
    resumen_preparacion.json conteos de cada paso (para el EDA y el reporte)

Uso (desde la raíz del proyecto):
    venv\\Scripts\\python.exe src\\preparar.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clientes import clave_cliente

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "full_2015_2026.parquet"
CATALOGO = RAIZ / "data" / "cat_articulos_202610010754.csv"
SALIDA = RAIZ / "data" / "processed"

INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2026-09-01")        # último mes completo; oct-2026 es parcial (LOG #3)
N_OBSOLETO = 12                         # meses sin venta para marcar obsoleto (§7)

VENTA = "Venta Producto"
DEVOLUCIONES = ["Devolución de producto", "Cargos por devolucion (vtas)", "Garantía de Producto",
                "Rotación de Inventarios"]
CLAVE_DUP = ["order_no", "factura", "id_articulo", "cantidad", "fechafactura", "subtotalmn", "concepto"]


def cargar():
    df = pd.read_parquet(ENTRADA)
    resumen = {"filas_crudas": len(df)}

    sin_origen = [c for c in df.columns if c not in ("source_file", "id_registro")]
    df = df.drop_duplicates(subset=sin_origen)
    resumen["tras_dup_completos"] = len(df)
    df = df.drop_duplicates(subset=CLAVE_DUP)
    resumen["tras_dup_clave"] = len(df)

    df = df[df.order_date < FIN + pd.offsets.MonthBegin(1)]
    resumen["tras_corte_oct2026"] = len(df)

    # desfase orden → factura (§6: cuantificar)
    dias = (df.fechafactura - df.order_date).dt.days
    resumen["pct_factura_distinta_orden"] = float((dias != 0).mean())
    resumen["desfase_max_dias"] = int(dias.max())
    return df, resumen


def familias():
    cat = pd.read_csv(CATALOGO, usecols=["id_articulo", "familia"])
    cat["familia"] = cat.familia.str.strip()
    return cat.set_index("id_articulo").familia


def anonimizar_clientes(df):
    """Clave de cliente (RFC/nombre, src/clientes.py) → código CL0001… ordenado por unidades vendidas.
    El RFC y el nombre no salen de aquí (§4)."""
    clave = clave_cliente(df)
    orden = df.cantidad.abs().groupby(clave).sum().sort_values(ascending=False).index
    codigos = pd.Series([f"CL{i + 1:04d}" for i in range(len(orden))], index=orden)
    return clave.map(codigos)


def lineas(df, fam):
    df = df[df.concepto.isin([VENTA] + DEVOLUCIONES)].copy()
    df["cliente"] = anonimizar_clientes(df)
    df["familia"] = df.id_articulo.map(fam).fillna("SIN_FAMILIA")
    df["mes"] = df.order_date.dt.to_period("M").dt.to_timestamp()
    cols = ["order_date", "mes", "order_no", "id_articulo", "familia", "id_cliente", "cliente", "concepto",
            "cantidad", "subtotalmn", "precioventaunidadmn", "precioventaunidadusd", "exchange_rate2"]
    return df[cols]


def precio_por_linea(v):
    """Cascada §9 a nivel línea. Regresa (precio MXN, origen).
    1. MXN directo (subtotal / cantidad, o el precio unitario MXN).
    2. USD × tipo de cambio de la línea (MXN por USD: la conversión correcta es multiplicar).
    3. USD × mediana mensual del tipo de cambio cuando falta el de la línea.
    El paso 4 (mediana del SKU en ±6 meses) se hace a nivel SKU-mes en precio_sku_mes()."""
    directo = (v.subtotalmn / v.cantidad).where(lambda s: s > 0)
    directo = directo.fillna(v.precioventaunidadmn.where(lambda s: s > 0))
    usd = v.precioventaunidadusd.where(lambda s: s > 0)
    fx_mes = v.groupby("mes").exchange_rate2.transform("median")
    fx_linea = usd * v.exchange_rate2
    fx_ref = usd * fx_mes
    precio = directo.fillna(fx_linea).fillna(fx_ref)
    origen = np.select([directo.notna(), fx_linea.notna(), fx_ref.notna()], ["mxn", "usd_fx", "usd_fx_mes"],
                       default="sin_precio")
    return precio, pd.Series(origen, index=v.index)


def precio_sku_mes(v):
    precio, origen = precio_por_linea(v)
    v = v.assign(precio=precio, origen=origen)
    ok = v[v.precio.notna()]
    pm = (ok.assign(imp=ok.precio * ok.cantidad).groupby(["id_articulo", "mes"])
          .agg(imp=("imp", "sum"), u=("cantidad", "sum"), origen=("origen", lambda s: s.mode().iloc[0])))
    pm["precio_mxn"] = pm.imp / pm.u
    pm = pm[["precio_mxn", "origen"]]

    # paso 4: SKU-mes con venta pero sin ningún precio → mediana del mismo SKU en ±6 meses
    con_venta = v.groupby(["id_articulo", "mes"]).size().index
    faltan = con_venta.difference(pm.index)
    rellenos = []
    for sku, mes in faltan:
        p = pm.loc[sku].precio_mxn if sku in pm.index.get_level_values(0) else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        cerca = p[(p.index >= mes - pd.DateOffset(months=6)) & (p.index <= mes + pd.DateOffset(months=6))]
        rellenos.append((sku, mes, cerca.median() if len(cerca) else np.nan, "sku_6m" if len(cerca) else "sin_precio"))
    if rellenos:
        r = pd.DataFrame(rellenos, columns=["id_articulo", "mes", "precio_mxn", "origen"]).set_index(["id_articulo", "mes"])
        pm = pd.concat([pm, r])
    return pm.sort_index().reset_index()


def panel(v, fam):
    """Rejilla SKU × mes (§7). Vida activa = [primer mes con venta, min(último mes con venta + N_OBSOLETO, FIN)].
    Los ceros posteriores a la última venta (hasta N_OBSOLETO meses) son la evidencia de que el SKU se apagó;
    más allá ya no se rellena para no inventar intermitencia (LOG_DECISIONES)."""
    u = v.groupby(["id_articulo", "mes"]).cantidad.sum()
    vida = u.reset_index().groupby("id_articulo").mes.agg(primera_venta="min", ultima_venta="max")
    vida["fin_vida"] = (vida.ultima_venta + pd.DateOffset(months=N_OBSOLETO)).clip(upper=FIN)

    meses = pd.date_range(INICIO, FIN, freq="MS")
    filas = [(sku, m) for sku, r in vida.iterrows() for m in meses[(meses >= r.primera_venta) & (meses <= r.fin_vida)]]
    p = pd.DataFrame(filas, columns=["id_articulo", "mes"])
    p = p.merge(u.rename("unidades").reset_index(), how="left").fillna({"unidades": 0})
    p = p.merge(vida.reset_index(), on="id_articulo")
    p["familia"] = p.id_articulo.map(fam).fillna("SIN_FAMILIA")
    p["edad_meses"] = ((p.mes.dt.year - p.primera_venta.dt.year) * 12 + p.mes.dt.month - p.primera_venta.dt.month)
    # obsoleto "hoy" (a FIN) solo para reporte; en el backtest se recalcula en cada origen (LOG #6)
    p["obsoleto_a_fin"] = p.ultima_venta <= FIN - pd.DateOffset(months=N_OBSOLETO)
    cols = ["id_articulo", "familia", "mes", "unidades", "primera_venta", "ultima_venta", "fin_vida", "edad_meses",
            "obsoleto_a_fin"]
    return p[cols].sort_values(["id_articulo", "mes"]).reset_index(drop=True)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df, resumen = cargar()
    fam = familias()
    lin = lineas(df, fam)

    ventas = lin[(lin.concepto == VENTA)]
    resumen["venta_lineas_cantidad_no_positiva"] = int((ventas.cantidad <= 0).sum())
    ventas = ventas[ventas.cantidad > 0].drop(columns="concepto")
    dev = lin[lin.concepto.isin(DEVOLUCIONES)].assign(unidades_dev=lambda d: -d.cantidad)

    pm = precio_sku_mes(ventas)
    pan = panel(ventas, fam)

    resumen.update({
        "venta_lineas": len(ventas), "venta_unidades": float(ventas.cantidad.sum()),
        "devolucion_lineas": len(dev), "devolucion_unidades": float(dev.unidades_dev.sum()),
        "skus": int(pan.id_articulo.nunique()), "skus_obsoletos_a_fin": int(pan.groupby("id_articulo").obsoleto_a_fin.first().sum()),
        "familias": int(pan.familia.nunique()), "clientes": int(ventas.cliente.nunique()),
        "clientes_id_crudo": int(ventas.id_cliente.nunique()),
        "panel_filas": len(pan), "panel_pct_ceros": float((pan.unidades == 0).mean()),
        "precio_origen": pm.origen.value_counts().to_dict(),
        "rango": [str(ventas.mes.min().date()), str(ventas.mes.max().date())],
    })

    ventas.drop(columns=["precioventaunidadusd", "exchange_rate2"]).to_parquet(SALIDA / "ventas.parquet", index=False)
    dev.drop(columns=["precioventaunidadusd", "exchange_rate2", "precioventaunidadmn"]).to_parquet(
        SALIDA / "devoluciones.parquet", index=False)
    pm.to_parquet(SALIDA / "precio_sku_mes.parquet", index=False)
    pan.to_parquet(SALIDA / "panel.parquet", index=False)
    (SALIDA / "resumen_preparacion.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
