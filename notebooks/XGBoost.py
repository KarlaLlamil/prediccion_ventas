"""XGBoost por serie (archivo del usuario) adaptado al proyecto (LOG #33).

Se conserva la idea original: un XGBRegressor por serie (mismos hiperparámetros), con features = mes + 3 lags,
métricas wMAPE / sMAPE / MAE / RMSE / R² y pronóstico del siguiente periodo para el top de productos.
Adaptaciones:
- Datos: `data/processed/panel.parquet` en lugar de SQLite. El panel ya es `concepto = 'Venta Producto'` por mes de
  `order_date` (no `fechafactura`), deduplicado y hasta sep-2026 (oct-2026 se eliminó, LOG #3).
- Horizonte h = 4 (lead time 3 + revisión 1): los lags son 4, 5 y 6 (con h = 4 no se conoce t−1…t−3).
- Top 30 (en lugar de 50) por unidades del último año, en dos niveles: SKU y familia.
- Además del ajuste in-sample del original, se evalúa fuera de muestra con el mismo walk-forward que el resto de
  los modelos (`backtest.py`: en cada origen T se entrena con datos ≤ T y se pronostica T + 4). Así se compara
  contra los nuestros en igualdad de condiciones. El in-sample se conserva solo como referencia: mide qué tan bien
  el modelo memoriza la historia, no qué tan bien pronostica.

Uso (desde la raíz):  venv\\Scripts\\python.exe notebooks\\XGBoost.py
Escribe data/processed/resultados/xgb_top30_{sku,familia}.parquet, xgb_top30_insample.csv y xgb_top30_futuro.csv.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from backtest import ORIGENES, SALIDA
from features import H, MESES, PROC

warnings.filterwarnings('ignore')
TOP = 30
LAGS = [H, H + 1, H + 2]                       # lag_4, lag_5, lag_6
FEATURES = ['mes'] + [f'lag_{i}' for i in LAGS]


# 1. Funciones de error robustas
def wmape(y_true, y_pred):
    suma_reales = np.sum(np.abs(y_true))
    if suma_reales == 0: return 0
    return np.sum(np.abs(y_true - y_pred)) / suma_reales

def smape(y_true, y_pred):
    denominador = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominador = np.where(denominador == 0, 1e-10, denominador)
    return np.mean(np.abs(y_true - y_pred) / denominador)


# 2. Extracción de datos (top 30 por unidades del último año, por SKU y por familia)
def extraer(nivel):
    panel = pd.read_parquet(PROC / "panel.parquet", columns=["id_articulo", "familia", "mes", "unidades"])
    fecha_corte = panel.mes.max() - pd.DateOffset(years=1)
    top = panel[panel.mes > fecha_corte].groupby(nivel).unidades.sum().nlargest(TOP).index
    df = panel[panel[nivel].isin(top)].groupby([nivel, "mes"], as_index=False).unidades.sum()
    return df.rename(columns={nivel: "serie", "mes": "ds", "unidades": "y"}).sort_values(["serie", "ds"])


# 3. Ingeniería de características (Feature Engineering)
def preparar_datos_supervisados(df_producto, lags=LAGS):
    # Asegurar que no falten meses en la serie (rellenar con 0 si no hubo ventas), hasta el último mes del panel
    rango_fechas = pd.date_range(start=df_producto['ds'].min(), end=MESES[-1], freq='MS')
    df_completo = pd.DataFrame({'ds': rango_fechas})
    df_completo = pd.merge(df_completo, df_producto[['ds', 'y']], on='ds', how='left').fillna({'y': 0})

    # Crear variables de fecha
    df_completo['mes'] = df_completo['ds'].dt.month

    # Crear lags (t-4, t-5, t-6): con h = 4 son los más recientes conocidos al pronosticar
    for i in lags:
        df_completo[f'lag_{i}'] = df_completo['y'].shift(i)

    # Eliminar las primeras filas que tienen NaN debido al shift()
    return df_completo.dropna().reset_index(drop=True)


def nuevo_modelo():
    return xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, objective='reg:squarederror',
                            n_jobs=1, random_state=0)


# 4. Entrenamiento y evaluación
def walk_forward(df_prep):
    """En cada origen T: entrena con objetivos ≤ T y pronostica T + 4 (sus lags 4–6 son T, T−1, T−2)."""
    filas = []
    for periodo, origenes in ORIGENES.items():
        for o in origenes:
            tr = df_prep[df_prep.ds <= MESES[o]]
            te = df_prep[df_prep.ds == MESES[o + H]]
            if len(tr) < 5 or te.empty:          # requiere un mínimo de datos tras crear los lags
                continue
            m = nuevo_modelo().fit(tr[FEATURES], tr.y)
            filas.append({"periodo": periodo, "origen": MESES[o], "mes_obj": MESES[o + H],
                          "XGBoost": max(0, float(m.predict(te[FEATURES])[0]))})
    return filas


def correr(nivel):
    df = extraer(nivel)
    resultados_metricas, predicciones_futuras, backtest = [], [], []
    for producto in df['serie'].unique():
        df_prep = preparar_datos_supervisados(df[df['serie'] == producto])
        if len(df_prep) < 5:
            continue

        # Ajuste in-sample, como el original (solo referencia)
        X, y = df_prep[FEATURES], df_prep['y']
        modelo_xgb = nuevo_modelo().fit(X, y)
        y_pred = np.maximum(0, modelo_xgb.predict(X))   # evitar predicciones negativas
        resultados_metricas.append({
            nivel: producto, 'R2': r2_score(y, y_pred), 'wMAPE': wmape(y, y_pred), 'sMAPE': smape(y, y_pred),
            'MAE': mean_absolute_error(y, y_pred), 'RMSE': np.sqrt(mean_squared_error(y, y_pred))})

        # 5. Predecir t + 4 (ene-2027) desde el último mes con datos (sep-2026)
        ultimo = df_prep['ds'].max()
        siguiente_mes = ultimo + pd.DateOffset(months=H)
        serie_y = df_prep.set_index('ds').y
        X_futuro = pd.DataFrame([[siguiente_mes.month] + [serie_y[ultimo - pd.DateOffset(months=i - H)] for i in LAGS]],
                                columns=FEATURES)
        predicciones_futuras.append({nivel: producto, 'fecha_pronostico': siguiente_mes,
                                     'cantidad_proyectada': max(0, float(modelo_xgb.predict(X_futuro)[0]))})

        # Fuera de muestra: walk-forward
        backtest += [{nivel: producto, **f} for f in walk_forward(df_prep)]
    return pd.DataFrame(resultados_metricas), pd.DataFrame(predicciones_futuras), pd.DataFrame(backtest)


def main():
    for nivel, nombre in [("id_articulo", "sku"), ("familia", "familia")]:
        df_metricas_xgb, df_pronostico_xgb, df_bt = correr(nivel)
        df_bt.to_parquet(SALIDA / f"xgb_top30_{nombre}.parquet", index=False)
        df_metricas_xgb.assign(nivel=nombre).to_csv(SALIDA / f"xgb_top30_insample_{nombre}.csv", index=False)
        df_pronostico_xgb.assign(nivel=nombre).to_csv(SALIDA / f"xgb_top30_futuro_{nombre}.csv", index=False)

        print(f"\n===== Top {TOP} por {nombre} =====")
        print("Métricas de ajuste in-sample XGBoost (top 30):")
        print(df_metricas_xgb.head(30).round(3).to_string(index=False))
        print(f"\nPredicción para {MESES[-1] + pd.DateOffset(months=H):%b-%Y} (top 30):")
        print(df_pronostico_xgb.head(30).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
