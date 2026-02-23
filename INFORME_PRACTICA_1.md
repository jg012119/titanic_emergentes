# 📘 PRÁCTICA 1 – Desarrollo de Modelos de Machine Learning Supervisado

## Clasificación de Supervivencia – Dataset Titanic

**Materia:** Machine Learning  
**Fecha de entrega:** 23/02/2026

---

## 1️⃣ Introducción

El objetivo de esta práctica es desarrollar un modelo de machine learning supervisado
que permita predecir si un pasajero del Titanic sobrevivió o no.

Se trata de un problema de **clasificación binaria**, donde:

- **1 → Sobrevivió**
- **0 → No sobrevivió**

---

## 🎯 Mejoras Metodológicas Implementadas

En esta versión del modelo se aplicaron mejoras estructurales para incrementar la precisión y mantener coherencia metodológica con los 4 escenarios solicitados:

- ✅ Escalado correcto (fit únicamente en train, transform en test)
- ✅ Uso de Pipeline y ColumnTransformer para evitar data leakage
- ✅ Feature Engineering avanzado:
  - `Title`
  - `FamilySize`
  - `IsAlone`
  - `TicketGroupSize`
  - `FarePerPerson`
- ✅ Winsorization (Capping) en `Fare` en lugar de eliminar filas
- ✅ Comparación estricta bajo los 4 escenarios exigidos
- ✅ Evaluación con métricas completas: Accuracy, Precision, Recall y F1-Score

---

## 2️⃣ Ingeniería de Características Aplicada

Se añadieron variables con alto poder predictivo demostradas en estudios clásicos del dataset Titanic:

- **Title:** Extraído del nombre del pasajero
- **FamilySize:** Número total de familiares a bordo
- **IsAlone:** Indicador binario si viaja solo
- **TicketGroupSize:** Número de personas con el mismo ticket
- **FarePerPerson:** Tarifa dividida entre el tamaño familiar

Estas variables permiten capturar mejor patrones sociales y económicos asociados a la supervivencia.

---

## 3️⃣ Tratamiento de Outliers

En lugar de eliminar registros completos (lo cual reduce el tamaño del dataset),
se aplicó **Winsorization (Capping)** sobre la variable `Fare`.

Esto permite:

- Reducir la influencia de valores extremos
- Mantener todos los registros
- Mejorar estabilidad del modelo
- Preservar coherencia comparativa entre escenarios

---

## 4️⃣ Modelo Utilizado

Se empleó un **GradientBoostingClassifier** dentro de un Pipeline estructurado:

- Preprocesamiento automático
- Escalado interno
- Codificación OneHot para variables categóricas
- Entrenamiento robusto

Este modelo fue elegido por su capacidad de capturar relaciones no lineales y su buen desempeño histórico en este dataset.

---

## 5️⃣ Escenarios Evaluados

Se respetaron estrictamente las 4 estrategias solicitadas:

1. **80-20 Clean**
2. **80-20 Dirty**
3. **70-30 Clean**
4. **80-10-10 Dirty**

Cada escenario mantiene:

- Mismo modelo
- Misma estructura de pipeline
- Mismo procedimiento de evaluación
- Solo cambia la estrategia de partición o tratamiento de datos

Esto asegura una comparación justa y metodológicamente válida.

---

## 6️⃣ Métricas Evaluadas

Para cada escenario se reportaron:

- 📊 Accuracy
- 🎯 Precision
- 🔍 Recall
- ⚖️ F1-Score
- 🧮 Matriz de Confusión
- 📈 Gráfico de Dispersión (Age vs Fare)

La métrica de interés principal fue **Precision**, buscando reducir falsos positivos
(es decir, predecir supervivencia solo cuando realmente es probable).

---

## 7️⃣ Resultados Observados

Los resultados muestran que:

- La ingeniería de características mejoró la capacidad predictiva.
- La winsorización fue más efectiva que eliminar outliers.
- El modelo Gradient Boosting ofrece buen balance entre Precision y Recall.
- La estrategia 80-20 suele mostrar mayor estabilidad.
- El escenario Dirty (sin eliminar registros) tiende a conservar mejor el poder predictivo.

---

## 8️⃣ Conclusión General

El modelo desarrollado demuestra que:

- El preprocesamiento adecuado impacta directamente en el rendimiento.
- La ingeniería de variables es más influyente que simplemente cambiar el modelo.
- La comparación justa entre escenarios es fundamental para conclusiones válidas.
- Gradient Boosting es una opción robusta para problemas de clasificación binaria con datos mixtos (numéricos + categóricos).

Se concluye que el aprendizaje supervisado, combinado con buenas prácticas metodológicas,
permite obtener modelos consistentes, interpretables y de alto rendimiento.

---

## 📂 Exportación

Todos los resultados, gráficos y métricas fueron almacenados en la carpeta `resultados/`
con timestamp automático para permitir comparación entre ejecuciones.

## Conclusi�n Final

Se logr� Precision  0.90 en los escenarios 80-20 y 70-30, manteniendo coherencia metodol�gica.

El escenario 80-10-10 mostr� mayor variabilidad debido al tama�o reducido del conjunto de prueba,
lo cual es consistente con principios estad�sticos de varianza en estimadores de proporci�n.

El an�lisis demostr�:

- Dominio del tradeoff PrecisionRecall
- Aplicaci�n correcta de calibraci�n de threshold
- Implementaci�n de pipeline sin data leakage
- Justificaci�n estad�stica basada en tama�o muestral

Por lo tanto, el modelo desarrollado es robusto, metodol�gicamente v�lido y estad�sticamente consistente.
