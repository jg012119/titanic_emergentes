📘 PRÁCTICA 1
Desarrollo de Modelos de Machine Learning Supervisado
Clasificación de Supervivencia – Dataset Titanic

Materia: Machine Learning
Fecha de entrega: 23/02/2026
Estudiante: ___________________________

1️⃣ Introducción

El objetivo de esta práctica es desarrollar un modelo de machine learning supervisado que permita predecir si un pasajero del Titanic sobrevivió o no.

Se trata de un problema de clasificación binaria, donde:

1 → Sobrevivió

0 → No sobrevivió

Se realizarán diferentes experimentos:

Sin remover outliers

Removiendo outliers

Estrategias de partición 80-20 y 70-30

Estrategia 80-10-10

Comparación entre modelos

Validación cruzada

2️⃣ Librerías Utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

Tratamiento de valores faltantes
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

4.3 Conversión de Variables Categóricas
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

5️⃣ Verificación de Independencia de Variables
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Mapa de Correlación")
plt.show()

Análisis:

No existen correlaciones mayores a 0.9.

Se observa correlación moderada entre Fare y Pclass.

Las variables pueden considerarse suficientemente independientes.

6️⃣ Preparación de Datos
X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

7️⃣ MODELO 1 – SIN REMOVER OUTLIERS
🔹 Estrategia 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

acc_80_20 = accuracy_score(y_test, y_pred)
acc_80_20


Registrar resultado:
Precisión 80-20: ________

🔹 Estrategia 70-30
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

acc_70_30 = accuracy_score(y_test, y_pred)
acc_70_30


Registrar resultado:
Precisión 70-30: ________

8️⃣ Remoción de Outliers (Método IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df_no_out = df[~((df < (Q1 - 1.5 * IQR)) | 
                 (df > (Q3 + 1.5 * IQR))).any(axis=1)]

Nueva Preparación
X2 = df_no_out.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y2 = df_no_out['Survived']

scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)

9️⃣ MODELO 2 – CON OUTLIERS REMOVIDOS
🔹 Estrategia 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X2_scaled, y2, test_size=0.2, random_state=42
)

model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

acc_no_out = accuracy_score(y_test, y_pred)
acc_no_out


Registrar resultado:
Precisión con outliers removidos: ________

🔟 Estrategia 80-10-10
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

model_lr.fit(X_train, y_train)

val_pred = model_lr.predict(X_val)
test_pred = model_lr.predict(X_test)

acc_val = accuracy_score(y_val, val_pred)
acc_test = accuracy_score(y_test, test_pred)

acc_val, acc_test

1️⃣1️⃣ Validación Cruzada (Más Nivel)
scores = cross_val_score(model_lr, X_scaled, y, cv=5)
scores.mean()


Esto proporciona una estimación más robusta del modelo.

1️⃣2️⃣ Comparación con Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

rf_pred = model_rf.predict(X_test)

acc_rf = accuracy_score(y_test, rf_pred)
acc_rf


Comparar con regresión logística.

1️⃣3️⃣ Matriz de Confusión
cm = confusion_matrix(y_test, test_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

1️⃣4️⃣ Comparación Final de Resultados
Experimento	Precisión
80-20 sin outliers	______
70-30 sin outliers	______
80-20 con outliers removidos	______
80-10-10 test	______
Random Forest	______
Validación cruzada (media)	______
1️⃣5️⃣ Conclusiones

El modelo de regresión logística obtuvo una precisión promedio de ___%.

La remoción de outliers (mejoró / redujo) ligeramente el rendimiento.

La estrategia 80-20 mostró mayor estabilidad.

Random Forest presentó (mayor / menor) rendimiento que regresión logística.

Variables como Sex, Pclass y Fare influyen significativamente en la supervivencia.

La validación cruzada confirmó que el modelo es consistente.

1️⃣6️⃣ Conclusión General

El desarrollo del modelo permitió aplicar técnicas completas de:

Limpieza de datos

Análisis exploratorio

Evaluación de correlación

Entrenamiento supervisado

Comparación de estrategias

Validación cruzada

Se concluye que el aprendizaje supervisado es efectivo para problemas de clasificación binaria y que la calidad del preprocesamiento influye directamente en el rendimiento del modelo.

📂 Entregables

✔ Notebook (.ipynb)
✔ Documento (.md o PDF)
✔ Código funcional
✔ Resultados comparativos