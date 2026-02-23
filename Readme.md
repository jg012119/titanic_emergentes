# 📘 Práctica 1 – Machine Learning Supervisado

## Clasificación de Supervivencia – Dataset Titanic

**Materia:** Machine Learning  
**Fecha de entrega:** 23/02/2026

---

## 📋 Descripción

Desarrollo de un modelo de Machine Learning supervisado para predecir la supervivencia de pasajeros del Titanic. Se implementa un clasificador binario usando **GradientBoosting** con optimización de threshold para maximizar **Precision ≥ 85%**.

## 🎯 Objetivo

Predecir si un pasajero sobrevivió o no (`Survived`: 1 = Sí, 0 = No) utilizando variables como edad, sexo, clase, tarifa, entre otras.

## 📁 Estructura del Proyecto

```
├── Titanic-Dataset.csv          # Dataset original (891 registros)
├── titanic_ml.py                # Script principal (Python)
├── .gitignore                   # Archivos ignorados por Git
├── README.md                    # Este archivo
└── resultados/                  # Resultados de las ejecuciones
    └── YYYY-MM-DD_HH-MM_FINAL/
        ├── matrices_confusion.png    # Matrices de confusión (4 escenarios)
        ├── dispersion_age_fare.png   # Gráfico de dispersión Age vs Fare
        ├── metricas_modelo.png       # Métricas comparativas entre escenarios
        ├── conclusiones.txt          # Conclusiones automáticas
        └── resultados.csv            # Tabla de resultados numéricos
```

## 🔬 Metodología

### 1. Análisis Exploratorio (EDA)

- Verificación de independencia de variables (matriz de correlación)
- No se encontraron correlaciones extremas (|r| > 0.8) entre predictores

### 2. Feature Engineering

- **Title**: Título extraído del nombre del pasajero
- **FamilySize**: Tamaño de la familia (SibSp + Parch + 1)
- **IsAlone**: Si viaja solo o no
- **TicketGroupSize**: Cantidad de pasajeros por ticket
- **FarePerPerson**: Tarifa dividida entre tamaño familiar

### 3. Tratamiento de Outliers

- **Winsorización (Capping)**: Limita valores extremos usando IQR sin eliminar filas
- Se aplica a las variables `Fare` y `Age`

### 4. Modelo

- **GradientBoostingClassifier** con Pipeline de preprocesamiento
- StandardScaler para variables numéricas
- OneHotEncoder para variables categóricas
- **Threshold optimization**: Se ajusta el umbral de decisión para maximizar Precision

## 📊 Escenarios Experimentales

| #   | Escenario             | Split                          | Outliers   |
| --- | --------------------- | ------------------------------ | ---------- |
| 1   | 80-20 Sin Outliers    | 80% train / 20% test           | Winsorized |
| 2   | 80-20 Con Outliers    | 80% train / 20% test           | Originales |
| 3   | 70-30 Sin Outliers    | 70% train / 30% test           | Winsorized |
| 4   | 80-10-10 Con Outliers | 80% train / 10% val / 10% test | Originales |

## 📈 Gráficos Generados

1. **Matrices de Confusión** – Los 4 escenarios en una sola figura (2×2)
2. **Gráfico de Dispersión** – Age vs Fare, coloreado por aciertos y errores
3. **Métricas del Modelo** – Barras comparativas (Accuracy, Precision, Recall, F1)

## 🚀 Ejecución

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Ejecutar

```bash
python titanic_ml.py
```

Los resultados se guardan automáticamente en la carpeta `resultados/` con un timestamp.

## 📝 Conclusiones

Las conclusiones se generan **automáticamente** al ejecutar el script, basándose en los resultados reales de cada ejecución. Se guardan en `conclusiones.txt` dentro de la carpeta de resultados.

**Hallazgos principales:**

- La **winsorización** mantiene todas las filas del dataset sin perder datos
- El split **80-20** proporciona un buen balance entre entrenamiento y evaluación
- El split **80-10-10** tiene mayor varianza por el tamaño reducido del test set
- El **threshold de decisión** permite ajustar el balance Precision vs Recall
- Se logró **Precision ≥ 85%** en todos los escenarios experimentales

## 🛠️ Tecnologías

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (GradientBoostingClassifier, Pipeline, ColumnTransformer)
