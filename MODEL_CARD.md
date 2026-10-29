# Ficha técnica del modelo (Model Card): PD B2B a 12 meses

> Documento técnico, metodológico y funcional del modelo, en el formato que suelen pedir los comités de
> modelos. **Datos sintéticos:** el contenido ilustra la estructura de la documentación, no un modelo
> aprobado por ninguna entidad.

## 1. Propósito y uso

| Campo | Detalle |
|---|---|
| Objetivo | Estimar la probabilidad de default (90+ días de atraso) en los 12 meses siguientes a una solicitud de línea de crédito B2B |
| Uso previsto | Admisión: aprobar, aprobar con límite reducido, enviar a revisión o rechazar; además, sugerir el límite |
| Usuarios | Analistas de crédito y cobranzas; área comercial (consulta vía API) |
| Fuera de alcance | Personas naturales, refinanciamientos y clientes con menos de 3 meses de historial de pagos |
| Responsable | Rosy Napuchi |

## 2. Datos

| Campo | Detalle |
|---|---|
| Fuente | Generador sintético `data/generate.py` (semilla 11) |
| Tamaño | 9,000 solicitudes; 8.5 % de default |
| Partición | 75 % entrenamiento (con 20 % interno para early stopping) y 25 % de prueba, estratificado |
| Variable objetivo | `default_12m` (1 = 90+ días de atraso en 12 meses) |
| Exclusiones | Ninguna en la demo. En producción: fraudes confirmados y casos sin ventana de desempeño completa |

## 3. Variables

| Variable | Definición | Relación esperada con el riesgo |
|---|---|---|
| `avg_days_late` | Días promedio de atraso en facturas (12 m) | ↑ riesgo (monótona) |
| `pct_invoices_late` | % de facturas pagadas tarde | ↑ (monótona) |
| `credit_utilization` | Saldo / línea aprobada | ↑ (monótona) |
| `bureau_score` | Puntaje de buró (simulado) | ↓ (monótona) |
| `bureau_delinquencies` | Morosidades reportadas | ↑ (monótona) |
| `dso_trend_days` | Cambio del DSO en los últimos 3 meses | ↑ (monótona) |
| `years_operating` | Antigüedad de la empresa | ↓ (monótona) |
| `log_revenue` | Log de ventas anuales | ↓ (monótona) |
| `requested_pct_revenue` | Monto solicitado / ventas × 100 | ↑ (monótona) |
| `sector_*` | Dummies de sector (base: Industria) | Sin restricción |

## 4. Metodología

- **Benchmark:** regresión logística con variables estandarizadas.
- **Modelo productivo:** XGBoost con restricciones monótonas en las 9 variables numéricas (98 árboles, profundidad 3, early stopping). Las restricciones garantizan que la dirección del efecto coincida con la tabla anterior, y `tests/test_model_and_api.py` lo verifica.
- **Escala de puntaje:** 600 puntos = odds 30:1; cada 20 puntos se duplican las odds (PDO).
- **Explicabilidad:** SHAP TreeExplainer. La API devuelve los 3 factores con mayor impacto en cada solicitud.

## 5. Desempeño (25 % de prueba)

| Métrica | Logística | XGBoost monótono |
|---|---|---|
| ROC-AUC | 0.836 | 0.830 |
| Gini | 0.672 | 0.660 |
| KS | 0.503 | 0.522 |
| Brier | 0.061 | 0.062 |
| PD media vs. default real | 8.7 % vs. 8.5 % | 8.5 % vs. 8.5 % |

Default observado por banda: A 0.0 % · B 2.1 % · C 4.3 % · D 12.3 % · E 33.8 %. El orden de riesgo es estricto entre bandas.

## 6. Plan de monitoreo (`python monitor.py`)

| Indicador | Frecuencia | Umbral | Acción |
|---|---|---|---|
| PSI del score | Mensual | < 0.10 estable · 0.10–0.25 vigilar · > 0.25 alerta | Alerta: análisis de causa y posible re-desarrollo |
| PSI por variable | Mensual | Mismos umbrales | Identificar qué variable cambió |
| Default observado vs. PD predicha (back-test) | Mensual, sobre cosechas con 12 meses de desempeño | Fuera del IC 95 % binomial | Recalibrar el intercepto |
| Gini | Trimestral | Caída > 10 puntos vs. desarrollo | Re-desarrollo |

**Simulación de tres escenarios:**

| Escenario | PSI del score | Calibración | Lectura |
|---|---|---|---|
| Mes estable | 0.003 (estable) | Calibrado | Sin acción |
| Cartera más riesgosa | 0.204 (vigilar) | Calibrado | Cambió la población, pero el modelo sigue ordenando y estimando bien |
| Shock macro no observado | 0.003 (estable) | **Subestima el riesgo** (8.5 % vs. 12.1 %) | El PSI no lo detecta; solo el back-test. Recalibrar |

**Conclusión de diseño:** el PSI por sí solo no basta. Hay que monitorear juntas la estabilidad de la población y la calibración contra el default real.

## 7. Limitaciones

- Los datos sintéticos siguen una relación logística: no representan las no linealidades de una cartera real.
- No hay validación fuera de tiempo con cosechas reales ni análisis de *reject inference*.
- El puntaje de buró es simulado y no proviene de ningún buró real.
- En un entorno regulado, el modelo requeriría validación independiente y aprobación de un comité antes de usarse.
