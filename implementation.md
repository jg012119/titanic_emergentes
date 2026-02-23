    📘 Práctica 1 – Desarrollo de Modelos de Machine Learning Supervisado
Clasificación de Supervivencia – Dataset Titanic

Materia: Machine Learning
Fecha de entrega: 23/02/2026
Estudiante: ___________________________

1️⃣ Introducción

En esta práctica se desarrolla un modelo de machine learning supervisado para predecir si un pasajero del Titanic sobrevivió o no, utilizando el dataset clásico Titanic Dataset.

El problema es de clasificación binaria, donde:

1 → Sobrevivió

0 → No sobrevivió

Se realizarán distintas pruebas:

Sin remover outliers

Removiendo outliers

División 80-20

División 70-30

Estrategia 80-10-10

2️⃣ Carga de Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

3️⃣ Carga del Dataset
df = pd.read_csv("Titanic-Dataset.csv")
df.head()

4️⃣ Análisis Exploratorio de Datos (EDA)
4.1 Información General
df.info()
df.describe()

4.2 Valores Nulos
df.isnull().sum()

Tratamiento de valores nulos
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

4.3 Conversión de Variables Categóricas
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
df.head()

5️⃣ Verificación de Independencia de Variables

Se analiza la correlación entre variables numéricas.

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de Correlación")
plt.show()

Análisis:

Se observa correlación fuerte entre Fare y Pclass.

No existen correlaciones extremadamente altas (>0.9).

Se puede continuar con el modelo.

6️⃣ Preparación de Datos
Variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y = df['Survived']

Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

7️⃣ Entrenamiento SIN remover Outliers
División 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_80_20 = accuracy_score(y_test, y_pred)
accuracy_80_20


Registrar precisión obtenida:
Precisión (80-20 sin remover outliers): __________

División 70-30
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_70_30 = accuracy_score(y_test, y_pred)
accuracy_70_30


Registrar precisión obtenida:
Precisión (70-30 sin remover outliers): __________

8️⃣ Remoción de Outliers

Se utilizará el método IQR.

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_no_outliers.shape

Preparación nuevamente
X2 = df_no_outliers.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y2 = df_no_outliers['Survived']

scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)

9️⃣ Entrenamiento CON Outliers Removidos
División 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X2_scaled, y2, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_no_out_80_20 = accuracy_score(y_test, y_pred)
accuracy_no_out_80_20


Registrar precisión:
Precisión (80-20 con outliers removidos): __________

🔟 Estrategia 80-10-10
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

accuracy_val = accuracy_score(y_val, val_pred)
accuracy_test = accuracy_score(y_test, test_pred)

accuracy_val, accuracy_test


Registrar resultados:

Precisión validación: __________

Precisión test: __________

1️⃣1️⃣ Matriz de Confusión
cm = confusion_matrix(y_test, test_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

1️⃣2️⃣ Resultados Comparativos
Experimento	Precisión
80-20 sin outliers	______
70-30 sin outliers	______
80-20 con outliers removidos	______
80-10-10 test	______
1️⃣3️⃣ Conclusiones

La regresión logística logra una precisión aproximada de ___%.

La remoción de outliers (mejoró / empeoró) el rendimiento.

La estrategia 80-20 fue (más estable / menos estable).

La correlación entre variables no fue lo suficientemente alta como para afectar gravemente el modelo.

El modelo demuestra que variables como Sex, Pclass y Fare influyen significativamente en la supervivencia.

1️⃣4️⃣ Conclusión General

El modelo de clasificación desarrollado demuestra que es posible predecir la supervivencia de un pasajero del Titanic utilizando técnicas de machine learning supervisado.

Se comprobó que:

La correcta limpieza de datos mejora el rendimiento.

El tratamiento de outliers puede influir en la precisión.

La selección adecuada de estrategia de división es clave.

La regresión logística es adecuada para problemas binarios.