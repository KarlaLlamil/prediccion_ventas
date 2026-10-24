"""Clave única de cliente a partir del RFC y del nombre (no del id_cliente).

El mismo cliente puede tener varios id_cliente en el tiempo (ej. hay clientes con 3 ids), y en 2023 el 93 % de
las filas de Postgres viene sin RFC. Por eso se unen las filas que comparten:
- un RFC válido (12–13 caracteres y que no sea genérico: XAXX010101000 = público en general, XEXX010101000 = extranjero), o
- el mismo nombre normalizado (sin acentos ni puntuación, sin "S.A. DE C.V." y similares, palabras en orden alfabético,
  para que "BAUTISTA HERNANDEZ MARIANO" = "MARIANO BAUTISTA HERNANDEZ").
Las uniones son transitivas: si A comparte RFC con B y B comparte nombre con C, los tres son el mismo cliente.

Uso:
    from clientes import clave_cliente
    df["cliente"] = clave_cliente(df)
"""
import re
import unicodedata

import pandas as pd

RFC_GENERICOS = {"XAXX010101000", "XEXX010101000"}
# Palabras de razón social que se quitan del final del nombre (S.A. DE C.V., S. DE R.L. DE C.V., S.A.P.I., S.C., A.C.…)
LEGALES = {"S", "A", "SA", "DE", "C", "V", "CV", "R", "L", "RL", "SRL", "SAPI", "SAB", "SAS", "SC", "AC", "P", "I", "B",
           "SPR", "RI", "MI", "SOFOM", "ENR", "ER"}


def normalizar_nombre(nombre):
    if pd.isna(nombre):
        return None
    n = unicodedata.normalize("NFKD", str(nombre).upper()).encode("ascii", "ignore").decode()
    palabras = re.sub(r"[^A-Z0-9&]+", " ", n).split()
    while palabras and palabras[-1] in LEGALES:
        palabras.pop()
    return " ".join(sorted(palabras)) or None


def rfc_valido(rfc):
    r = rfc.str.strip().str.upper()
    return r.where(r.str.len().isin([12, 13]) & ~r.isin(RFC_GENERICOS))


def clave_cliente(df):
    """Regresa una Serie (mismo índice que df) con la clave del cliente: el RFC válido más frecuente del grupo,
    o el nombre normalizado si el grupo nunca tuvo RFC válido."""
    rfc = rfc_valido(df.rfc)
    nombres = df.nombre_cliente.drop_duplicates()
    nombre = df.nombre_cliente.map(dict(zip(nombres, nombres.map(normalizar_nombre))))

    padre = {}
    def raiz(x):
        padre.setdefault(x, x)
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    pares = pd.DataFrame({"r": "rfc:" + rfc, "n": "nom:" + nombre}).drop_duplicates()
    for r, n in pares.itertuples(index=False):
        nodos = [x for x in (r, n) if pd.notna(x)]
        for x in nodos[1:]:
            padre[raiz(x)] = raiz(nodos[0])
        if nodos:
            raiz(nodos[0])

    nodo = ("rfc:" + rfc).fillna("nom:" + nombre)
    grupo = nodo.map({x: raiz(x) for x in nodo.dropna().unique()})
    # nombre legible de la clave: el RFC válido más frecuente del grupo; si no tiene, su nombre normalizado
    con_rfc = pd.DataFrame({"g": grupo, "rfc": rfc}).dropna()
    rfc_grupo = con_rfc.groupby("g").rfc.agg(lambda s: s.value_counts().index[0])
    return grupo.map(rfc_grupo).fillna(nombre).rename("cliente")
