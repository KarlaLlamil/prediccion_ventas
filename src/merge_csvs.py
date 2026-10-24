import re
import pandas as pd
import unicodedata
from pathlib import Path

# 1. Read headers first and compare
files = sorted(Path("../data/mensual_si/").glob("*.csv"))
big = Path("../data/ventas_detalle_si_202610010752.csv")
new_files = sorted(Path("../data/ultimos_meses/").glob("*.csv"))
SALIDA = Path("../data/full_2015_2026.parquet")   # parquet guarda los tipos; un CSV los perdería al volver a leerlo

# Tipos de la tabla ventas_detalle_si en Postgres (data/dataType.docx), con los nombres ya normalizados.
# int4 → "Int64" (entero que acepta vacíos), float8 → "float64", varchar → "string".
# Las fechas se leen como texto y se convierten después (ver convertir_fechas).
INT = ["id_articulo", "id_cliente", "id_descuento_global", "id_sucursal", "id_metasregion", "id_homologacion",
       "id_listaprecioesp", "mayorista", "so", "fechaano", "si", "order_no", "impuesto1", "tipoventa",
       "id_registro", "sublinea"]
FLOAT = ["cantidad", "tipocambio", "subtotalmn", "precioventaunidadmn", "subtotalusd", "precioventaunidadusd",
         "preciotipocambio", "exchange_rate2", "precio", "descuentoimporte", "wcost", "wcogs", "exchange_rate",
         "preciousd", "wcost_usd", "wcogs_usd"]
FECHAS = ["order_date", "fechafactura"]               # date
TIMESTAMPS = ["fechaemision", "fecharegistro"]       # timestamp
SCHEMA = {**{c: "Int64" for c in INT}, **{c: "float64" for c in FLOAT}}   # lo que no esté aquí → "string"

def read_csv_safe(path, **kwargs):
    """Try UTF-8, then Windows-1252 (a superset of Latin-1 for Western text)."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            print(f"{path.name}: read with {enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")

def normalize_col(name: str) -> str:
    """Lowercase, strip, remove accents, and replace spaces with underscores."""
    name = str(name).strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\s+", "_", name)
    return name.lower()

RENAMES = {
    # "old_name_in_2015": "new_name_in_2026",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [RENAMES.get(normalize_col(c), normalize_col(c)) for c in df.columns]
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate columns after normalization: {dupes}")
    return df

def read_typed(path):
    """Lee el CSV diciéndole a pandas el tipo de cada columna (por su nombre original en ese archivo).

    Si un valor no cabe en su tipo (ej. texto en una columna Int64) read_csv truena: mejor que convertirlo en silencio.
    """
    header = read_csv_safe(path, nrows=0).columns
    dtype = {c: SCHEMA.get(RENAMES.get(normalize_col(c), normalize_col(c)), "string") for c in header}
    return normalize_columns(read_csv_safe(path, dtype=dtype))

def convertir_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Los CSV mensuales traen '2026-05-19T11:55:55.283-06:00' y Postgres '2026-05-19 11:55:55.283'.

    Se quita la zona horaria y se deja la hora local de México, que es lo que guarda Postgres.
    """
    for c in FECHAS + TIMESTAMPS:
        df[c] = pd.to_datetime(df[c].str[:23].str.replace("T", " "), format="ISO8601")
    for c in FECHAS:
        df[c] = df[c].dt.normalize()
    return df

ref_cols = list(normalize_columns(read_csv_safe(big, nrows=0)).columns)
for f in files + new_files:
    cols = list(normalize_columns(read_csv_safe(f, nrows=0)).columns)
    if set(cols) !=set(ref_cols):
        print(f"MISMATCH in {f.name}")
        print("  missing:", set(ref_cols) - set(cols))
        print("  extra:  ", set(cols) - set(ref_cols))

# 2. Merge (add a column to trace the origin of each row)
dfs = []
for f in files + [big] + new_files:
    df = convertir_fechas(read_typed(f))
    df["source_file"] = f.name
    dfs.append(df)

# Traslape: la exportación de Postgres llega al 21-may-2026 y ultimos_meses/2026_05.csv trae el mes completo
# (incluye las mismas facturas del 4 al 21). Desde el primer mes de ultimos_meses manda ultimos_meses.
corte = min(d.fechafactura.min() for d in dfs[len(files) + 1:]).to_period("M").to_timestamp()
pg = dfs[len(files)]
print(f"Postgres: se quitan {(pg.fechafactura >= corte).sum():,} filas con fechafactura >= {corte:%Y-%m-%d} (traslape)")
dfs[len(files)] = pg[pg.fechafactura < corte]

merged = pd.concat(dfs, ignore_index=True)

# 3. Sanity checks
print(merged.shape)
print(merged.duplicated(subset=[c for c in merged.columns if c not in ("source_file", "id_registro")]).sum(),
      "fully duplicated rows (ignoring source_file / id_registro)")
print(merged.fechafactura.min(), merged.fechafactura.max())
print(merged.dtypes.value_counts())

merged.to_parquet(SALIDA, index=False)
