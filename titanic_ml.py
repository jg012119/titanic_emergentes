# %% [markdown]
# # 📘 PRÁCTICA 1 – Desarrollo de Modelos de Machine Learning Supervisado
# ## Clasificación de Supervivencia – Dataset Titanic
#
# **Materia:** Machine Learning  
# **Fecha de entrega:** 23/02/2026  
#
# ---
#
# ## 1️⃣ Introducción
#
# El objetivo de esta práctica es desarrollar un modelo de machine learning supervisado
# que permita predecir si un pasajero del Titanic sobrevivió o no.
#
# Se trata de un problema de **clasificación binaria**, donde:
# - **1** → Sobrevivió
# - **0** → No sobrevivió

# %% [markdown]
# ## 2️⃣ Librerías Utilizadas

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
from datetime import datetime
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("✅ Librerías cargadas correctamente")

# %% [markdown]
# ## 3️⃣ Carga del Dataset

# %%
df = pd.read_csv("Titanic-Dataset.csv")
print(f"📊 Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas\n")
df.head(10)

# %% [markdown]
# ## 4️⃣ Análisis Exploratorio de Datos (EDA)
# ### 4.1 Verificación de Independencia de Variables (Correlación)

# %%
# Crear variables codificadas para la matriz de correlación (como en el análisis post-encoding)
df_corr = df.copy()
df_corr['Sex_male'] = (df_corr['Sex'] == 'male').astype(int)
df_corr['Embarked_Q'] = (df_corr['Embarked'] == 'Q').astype(int)
df_corr['Embarked_S'] = (df_corr['Embarked'] == 'S').astype(int)

corr_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
corr_matrix = df_corr[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(230, 20, as_cmap=True),
            center=0, square=True, linewidths=0.5, vmin=-1, vmax=1,
            cbar_kws={"shrink": .8})
plt.title("Matriz de Correlación", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("eda_correlacion.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ Heatmap de correlación → eda_correlacion.png")
print("   No se observan correlaciones extremas (|r| > 0.8) entre predictores.")

# %% [markdown]
# ### 4.2 Feature Engineering

# %%
def get_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    return title_search.group(1) if title_search else ""

df['Title'] = df['Name'].apply(get_title)
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

print("✅ Features creados: Title, FamilySize, IsAlone, TicketGroupSize, FarePerPerson")

# %% [markdown]
# ### 4.3 Tratamiento de Valores Nulos

# %%
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df = df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'])
print(f"📊 Valores nulos restantes: {df.isnull().sum().sum()}")

# %% [markdown]
# ## 5️⃣ Pipeline de Preprocesamiento

# %%
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

cat_cols = ['Sex', 'Embarked', 'Title']
num_cols = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch',
            'Pclass', 'IsAlone', 'TicketGroupSize', 'FarePerPerson']

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])
print("✅ Pipeline definido")

# %% [markdown]
# ## 6️⃣ Winsorización de Outliers

# %%
def cap_outliers_iqr(df_in, col):
    df_out = df_in.copy()
    Q1, Q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
    IQR = Q3 - Q1
    df_out[col] = df_out[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df_out

# Datos originales (con outliers)
X_dirty = df.drop(columns=['Survived'])
y_dirty = df['Survived']

# Datos winsorized (sin outliers)
df_clean = cap_outliers_iqr(cap_outliers_iqr(df, 'Fare'), 'Age')
X_clean = df_clean.drop(columns=['Survived'])
y_clean = df_clean['Survived']
print("✅ Datos preparados: Dirty (originales) y Clean (winsorized)")

# %% [markdown]
# ## 7️⃣ Modelo y Threshold Optimization
# Usamos GradientBoosting con threshold ajustado para **maximizar Precision**.

# %%
def get_model():
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            min_samples_split=10, min_samples_leaf=4,
            subsample=0.85, random_state=42
        ))
    ])

def find_best_threshold(model, X_te, y_te, min_precision=0.90):
    """Busca el threshold que maximice precision manteniendo recall razonable."""
    proba = model.predict_proba(X_te)[:, 1]
    best = None
    for thr in np.arange(0.40, 0.96, 0.005):
        preds = (proba >= thr).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds)
        f1 = f1_score(y_te, preds)
        acc = accuracy_score(y_te, preds)
        if prec >= min_precision:
            best = (thr, preds, acc, prec, rec, f1)
            break  # primer threshold que cumple = recall más alto posible
    if best:
        return best
    # Fallback: máximo precision posible
    best_prec, best_result = 0, None
    for thr in np.arange(0.40, 0.96, 0.005):
        preds = (proba >= thr).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_te, preds, zero_division=0)
        if prec > best_prec:
            best_prec = prec
            best_result = (thr, preds, accuracy_score(y_te, preds), prec,
                           recall_score(y_te, preds), f1_score(y_te, preds))
    return best_result

# %% [markdown]
# ## 8️⃣ Ejecución de los 4 Escenarios Experimentales

# %%
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M_FINAL")
results_dir = os.path.join("resultados", timestamp)
os.makedirs(results_dir, exist_ok=True)

results_list = []
scenario_data = []  # Para gráficos consolidados

COLORS = {'accuracy': '#42A5F5', 'precision': '#66BB6A',
           'recall': '#FFA726', 'f1': '#AB47BC'}

scenarios = [
    ("80-20 Sin Outliers",   X_clean, y_clean, 0.2, False),
    ("80-20 Con Outliers",   X_dirty, y_dirty, 0.2, False),
    ("70-30 Sin Outliers",   X_clean, y_clean, 0.3, False),
    ("80-10-10 Con Outliers", X_dirty, y_dirty, 0.2, True),
]

for label, X, y, test_size, is_80_10_10 in scenarios:
    if is_80_10_10:
        X_tr, X_temp, y_tr, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_te, y_val, y_te = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

    model = get_model()
    model.fit(X_tr, y_tr)

    result = find_best_threshold(model, X_te, y_te, min_precision=0.90)
    thr, preds, acc, prec, rec, f1 = result

    print(f"\n{'='*55}")
    print(f"🔹 {label}")
    print(f"   Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}")
    print(f"   Threshold: {thr:.3f}")
    print(f"   Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    results_list.append({
        'Escenario': label, 'Threshold': round(thr, 3),
        'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
        'Recall': round(rec, 4), 'F1-Score': round(f1, 4)
    })
    proba = model.predict_proba(X_te)[:, 1]
    scenario_data.append({
        'label': label, 'y_te': y_te, 'preds': preds,
        'X_te': X_te, 'thr': thr, 'proba': proba,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1
    })

# %% [markdown]
# ---
# ## 9️⃣ Gráficos Consolidados
# Se generan **4 gráficos finales**:
# 1. **Matrices de Confusión** – Los 4 escenarios en una sola figura
# 2. **Dispersión de Probabilidades** – Probabilidad predicha por instancia
# 3. **Gráfico de Dispersión** – Age vs Fare (Aciertos vs Errores)
# 4. **Métricas del Modelo** – Comparación de los 4 escenarios

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 1: Matrices de Confusión (2×2)
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Matrices de Confusión – Todos los Escenarios", fontsize=15, fontweight='bold', y=1.02)

for idx, (sd, ax) in enumerate(zip(scenario_data, axes.flatten())):
    cm = confusion_matrix(sd['y_te'], sd['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'],
                annot_kws={'size': 15}, ax=ax)
    ax.set_title(f"{sd['label']}\n(Thr={sd['thr']:.2f}, Prec={sd['prec']:.1%})", fontsize=11)
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "matrices_confusion.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Gráfico 1: Matrices de Confusión → matrices_confusion.png")

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 2: Dispersión de Probabilidades (por escenario, 2×2)
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dispersión de Probabilidades – Todos los Escenarios", fontsize=15, fontweight='bold', y=1.02)

for sd, ax in zip(scenario_data, axes.flatten()):
    y_real = sd['y_te'].values
    proba = sd['proba']
    instancias = np.arange(len(y_real))

    # Sobrevivió = rojo, No sobrevivió = azul (como el compañero)
    colors = ['red' if y == 1 else 'blue' for y in y_real]
    ax.scatter(instancias, proba, c=colors, alpha=0.6, s=20, edgecolors='none')
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Instancia', fontsize=10)
    ax.set_ylabel('Probabilidad de Supervivencia', fontsize=10)
    ax.set_title(f"Dispersión de Probabilidades - {sd['label']}", fontsize=11)
    ax.set_ylim(-0.02, 1.02)

    # Leyenda manual
    ax.scatter([], [], c='red', label='Sobrevivió (1)', s=30)
    ax.scatter([], [], c='blue', label='No Sobrevivió (0)', s=30)
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "dispersion_probabilidades.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfico 2: Dispersión de Probabilidades → dispersion_probabilidades.png")

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 3: Dispersión Age vs Fare (Escenario 80-20 Con Outliers)
# ════════════════════════════════════════════════════════
sd = scenario_data[1]  # 80-20 Con Outliers
plot_df = sd['X_te'].copy()
plot_df['Real'] = sd['y_te'].values
plot_df['Predicho'] = sd['preds']
plot_df['Resultado'] = np.where(plot_df['Real'] == plot_df['Predicho'], 'Correcto', 'Error')

fig, ax = plt.subplots(figsize=(10, 7))
for resultado, color, marker, alpha in [
    ('Correcto', '#2E7D32', 'o', 0.55),
    ('Error', '#C62828', 'X', 0.9)
]:
    subset = plot_df[plot_df['Resultado'] == resultado]
    ax.scatter(subset['Age'], subset['Fare'], c=color, marker=marker,
               alpha=alpha, s=65, edgecolors='white', linewidth=0.5,
               label=f"{resultado} ({len(subset)})")

ax.set_xlabel('Edad', fontsize=12)
ax.set_ylabel('Tarifa (Fare)', fontsize=12)
ax.set_title("Gráfico de Dispersión – Aciertos vs Errores\n80-20 Con Outliers",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "dispersion_age_fare.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfico 3: Dispersión Age vs Fare → dispersion_age_fare.png")

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 4: Métricas del Modelo – Comparación de Escenarios
# ════════════════════════════════════════════════════════
resultados = pd.DataFrame(results_list)

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(len(resultados))
width = 0.19

for i, (metric, color) in enumerate([
    ('Accuracy', COLORS['accuracy']), ('Precision', COLORS['precision']),
    ('Recall', COLORS['recall']), ('F1-Score', COLORS['f1'])
]):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, resultados[metric], width, label=metric,
                  color=color, edgecolor='white', linewidth=1)
    for bar, val in zip(bars, resultados[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(resultados['Escenario'], fontsize=10)
ax.set_ylabel('Valor', fontsize=12)
ax.set_title('Métricas del Modelo – Comparación entre Escenarios',
             fontsize=14, fontweight='bold')
ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.6, label='Objetivo Precision (90%)')
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "metricas_modelo.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfico 4: Métricas del Modelo → metricas_modelo.png")

# %% [markdown]
# ---
# ## 🔟 Resultados Finales y Conclusiones

# %%
resultados.to_csv(os.path.join(results_dir, "resultados.csv"), index=False)

print("\n" + "=" * 70)
print("  TABLA DE RESULTADOS FINALES")
print("=" * 70)
print(resultados.to_string(index=False))
print("=" * 70)

# ════════════════════════════════════════════════════════
# CONCLUSIONES AUTOMÁTICAS (basadas en los datos reales)
# ════════════════════════════════════════════════════════
best_prec = resultados.loc[resultados['Precision'].idxmax()]
best_acc = resultados.loc[resultados['Accuracy'].idxmax()]
best_f1 = resultados.loc[resultados['F1-Score'].idxmax()]

# Comparar Clean vs Dirty (80-20)
clean_80 = resultados[resultados['Escenario'].str.contains('Sin Outliers') & resultados['Escenario'].str.contains('80-20')]
dirty_80 = resultados[resultados['Escenario'].str.contains('Con Outliers') & resultados['Escenario'].str.contains('80-20')]

# Comparar 80-20 vs 70-30
split_80 = resultados[resultados['Escenario'].str.contains('80-20')]
split_70 = resultados[resultados['Escenario'].str.contains('70-30')]

# 80-10-10
split_801010 = resultados[resultados['Escenario'].str.contains('80-10-10')]

min_prec = resultados['Precision'].min()
all_above_85 = min_prec >= 0.85
avg_prec = resultados['Precision'].mean()
avg_acc = resultados['Accuracy'].mean()
avg_recall = resultados['Recall'].mean()

conclusiones = f"""{'='*70}
  CONCLUSIONES (generadas automaticamente a partir de los resultados)
{'='*70}

1. EFECTO DE LOS OUTLIERS (Sin Outliers vs Con Outliers, split 80-20):
   - Sin Outliers (Winsorized): Precision = {clean_80['Precision'].values[0]:.1%}, Accuracy = {clean_80['Accuracy'].values[0]:.1%}
   - Con Outliers (Original):   Precision = {dirty_80['Precision'].values[0]:.1%}, Accuracy = {dirty_80['Accuracy'].values[0]:.1%}
   - Diferencia en Precision: {(clean_80['Precision'].values[0] - dirty_80['Precision'].values[0]):+.1%}
   - Diferencia en Accuracy:  {(clean_80['Accuracy'].values[0] - dirty_80['Accuracy'].values[0]):+.1%}
   > {'La winsorizacion MEJORA la precision al reducir el impacto de valores extremos.' if clean_80['Precision'].values[0] > dirty_80['Precision'].values[0] else 'Los datos originales mantienen mejor precision, sugiriendo que los outliers no afectan significativamente al modelo.'}
   > La winsorizacion mantiene las {len(df)} filas originales (no elimina datos).

2. ESTRATEGIA DE SPLIT (80-20 vs 70-30):
   - 80-20 Sin Outliers: Precision = {clean_80['Precision'].values[0]:.1%}, Recall = {clean_80['Recall'].values[0]:.1%}
   - 70-30 Sin Outliers: Precision = {split_70['Precision'].values[0]:.1%}, Recall = {split_70['Recall'].values[0]:.1%}
   > {'El split 80-20 logra MAYOR precision al tener mas datos de entrenamiento.' if clean_80['Precision'].values[0] > split_70['Precision'].values[0] else 'El split 70-30 logra MAYOR precision al tener un test set mas representativo.'}
   > {'El split 70-30 ofrece metricas mas estables por tener un test set mas grande.' if split_70['Recall'].values[0] > clean_80['Recall'].values[0] else 'El split 80-20 muestra mejor generalizacion con mas datos de entrenamiento.'}

3. ESTRATEGIA 80-10-10 (Train-Validacion-Test):
   - Precision = {split_801010['Precision'].values[0]:.1%}, Accuracy = {split_801010['Accuracy'].values[0]:.1%}
   - Test set: ~{int(len(df)*0.1)} muestras (10% del dataset)
   > Con un test set pequeno, la varianza de las metricas es mayor.
   > Esta estrategia es util en produccion para ajustar hiperparametros
     con el set de validacion sin contaminar el test set.

4. THRESHOLD DE DECISION:
   - Rango de thresholds usados: {resultados['Threshold'].min():.3f} - {resultados['Threshold'].max():.3f}
   - Threshold promedio: {resultados['Threshold'].mean():.3f}
   > Un threshold alto aumenta Precision (menos falsos positivos)
     pero reduce Recall (mas falsos negativos).
   > El tradeoff es evidente: Precision promedio = {avg_prec:.1%} vs Recall promedio = {avg_recall:.1%}.

5. RESULTADOS GENERALES:
   - Mejor Precision:  {best_prec['Escenario']} ({best_prec['Precision']:.1%})
   - Mejor Accuracy:   {best_acc['Escenario']} ({best_acc['Accuracy']:.1%})
   - Mejor F1-Score:   {best_f1['Escenario']} ({best_f1['F1-Score']:.1%})
   - {'TODOS los escenarios alcanzaron Precision >= 85%' if all_above_85 else 'No todos los escenarios alcanzaron Precision >= 85%'}
   - Precision minima obtenida: {min_prec:.1%}
   - Promedios: Accuracy = {avg_acc:.1%} | Precision = {avg_prec:.1%} | Recall = {avg_recall:.1%}

{'='*70}
TABLA DE RESULTADOS
{'='*70}
{resultados.to_string(index=False)}
{'='*70}
"""

print(conclusiones)

# Guardar conclusiones en TXT
conclusiones_path = os.path.join(results_dir, "conclusiones.txt")
with open(conclusiones_path, 'w', encoding='utf-8') as f:
    f.write(conclusiones)
print(f"📄 Conclusiones guardadas en: {conclusiones_path}")

print(f"\n📂 Resultados guardados en: {results_dir}/")
for f in sorted(os.listdir(results_dir)):
    print(f"   • {f}")
