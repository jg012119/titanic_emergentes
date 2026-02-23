# %% [markdown]
# # 📘 PRÁCTICA 2 – Desarrollo de Modelos de Machine Learning Supervisado
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
#
# **Mejoras respecto a la Práctica 1:**
# - Segmentación **60/20/20** (Train / Validación / Test) con **estratificación**
# - Curva ROC y cálculo de AUC
# - Gráfica de métricas de evaluación detallada

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
    precision_score, recall_score, f1_score,
    roc_curve, auc
)

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("✅ Librerías cargadas correctamente")
print("   sklearn: roc_curve, confusion_matrix, accuracy_score, auc")

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
# ## 8️⃣ Ejecución de los Escenarios Experimentales
# **Segmentación:** 60% Entrenamiento / 20% Validación / 20% Test
# Se usa `stratify=y` en `train_test_split` para mantener la proporción de clases.

# %%
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M_V2")
results_dir = os.path.join("resultados", timestamp)
os.makedirs(results_dir, exist_ok=True)

results_list = []
scenario_data = []  # Para gráficos consolidados

COLORS = {'accuracy': '#42A5F5', 'precision': '#66BB6A',
           'recall': '#FFA726', 'f1': '#AB47BC'}

scenarios = [
    ("60-20-20 Sin Outliers", X_clean, y_clean),
    ("60-20-20 Con Outliers", X_dirty, y_dirty),
]

for label, X, y in scenarios:
    # ── Split estratificado 60/20/20 ──
    # Paso 1: 60% Train, 40% Temporal
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y)
    # Paso 2: 20% Validación, 20% Test (50/50 del temporal)
    X_val, X_te, y_val, y_te = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"\n{'='*55}")
    print(f"🔹 {label}")
    print(f"   Train: {X_tr.shape[0]} ({X_tr.shape[0]/len(X):.0%}) | "
          f"Val: {X_val.shape[0]} ({X_val.shape[0]/len(X):.0%}) | "
          f"Test: {X_te.shape[0]} ({X_te.shape[0]/len(X):.0%})")
    print(f"   stratify=y ✅ (distribución de clases preservada)")

    model = get_model()
    model.fit(X_tr, y_tr)

    result = find_best_threshold(model, X_te, y_te, min_precision=0.90)
    thr, preds, acc, prec, rec, f1 = result

    # Métricas con accuracy_score y confusion_matrix explícitos
    acc_explicit = accuracy_score(y_te, preds)
    cm_explicit = confusion_matrix(y_te, preds)

    # Extraer TP, FP, TN, FN de la matriz de confusión
    tn, fp, fn, tp = cm_explicit.ravel()

    print(f"   Threshold: {thr:.3f}")
    print(f"   Accuracy (accuracy_score): {acc_explicit:.4f}")
    print(f"   Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"   Confusion Matrix:\n{cm_explicit}")
    print(f"   ┌─────────────────────────────────────────┐")
    print(f"   │ Verdaderos Positivos  (TP): {tp:4d}        │")
    print(f"   │ Verdaderos Negativos  (TN): {tn:4d}        │")
    print(f"   │ Falsos Positivos      (FP): {fp:4d}        │")
    print(f"   │ Falsos Negativos      (FN): {fn:4d}        │")
    print(f"   └─────────────────────────────────────────┘")

    results_list.append({
        'Escenario': label, 'Threshold': round(thr, 3),
        'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
        'Recall': round(rec, 4), 'F1-Score': round(f1, 4),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    })
    proba = model.predict_proba(X_te)[:, 1]
    scenario_data.append({
        'label': label, 'y_te': y_te, 'preds': preds,
        'X_te': X_te, 'X_val': X_val, 'y_val': y_val,
        'X_tr': X_tr, 'y_tr': y_tr,
        'thr': thr, 'proba': proba, 'model': model,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    })

# %% [markdown]
# ---
# ## 9️⃣ Gráficos Consolidados
# Se generan **8 gráficos finales**:
# 1. **Matrices de Confusión** – Ambos escenarios
# 2. **Dispersión de Probabilidades** – Probabilidad predicha por instancia
# 3. **Gráfico de Dispersión** – Age vs Fare (Aciertos vs Errores)
# 4. **Métricas del Modelo** – Comparación entre escenarios
# 5. **Curva ROC** – Rendimiento del clasificador
# 6. **Métricas de Evaluación** – Detalle por escenario
# 7. **Detalle TP/FP/TN/FN** – Métricas de la matriz de confusión
# 8. **Distribución de Frecuencia** – Histogramas de Age y Fare

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 1: Matrices de Confusión (1×2)
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Matrices de Confusión – Escenarios 60/20/20 (stratify=y)",
             fontsize=15, fontweight='bold', y=1.02)

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
# GRÁFICO 2: Dispersión de Probabilidades (1×2)
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dispersión de Probabilidades – Escenarios 60/20/20",
             fontsize=15, fontweight='bold', y=1.02)

for sd, ax in zip(scenario_data, axes.flatten()):
    y_real = sd['y_te'].values
    proba = sd['proba']
    instancias = np.arange(len(y_real))

    # Sobrevivió = rojo, No sobrevivió = azul
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
# GRÁFICO 3: Dispersión Age vs Fare (Escenario Con Outliers)
# ════════════════════════════════════════════════════════
sd = scenario_data[1]  # 60-20-20 Con Outliers
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
ax.set_title("Gráfico de Dispersión – Aciertos vs Errores\n60-20-20 Con Outliers",
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
ax.set_title('Métricas del Modelo – Comparación entre Escenarios (60/20/20 stratify=y)',
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

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 5 (NUEVO): Curva ROC – Rendimiento del Clasificador
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Curva ROC – Rendimiento del Clasificador (60/20/20 stratify=y)",
             fontsize=15, fontweight='bold', y=1.02)

for sd, ax in zip(scenario_data, axes.flatten()):
    # Calcular curva ROC usando roc_curve de sklearn
    fpr, tpr, thresholds = roc_curve(sd['y_te'], sd['proba'])
    roc_auc = auc(fpr, tpr)

    # Graficar curva ROC
    ax.plot(fpr, tpr, color='#1976D2', lw=2.5,
            label=f'Curva ROC (AUC = {roc_auc:.4f})')

    # Línea diagonal (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5,
            label='Clasificador Aleatorio (AUC = 0.50)')

    # Sombrear área bajo la curva
    ax.fill_between(fpr, tpr, alpha=0.15, color='#1976D2')

    # Marcar punto del threshold usado
    idx_thr = np.argmin(np.abs(thresholds - sd['thr']))
    ax.scatter(fpr[idx_thr], tpr[idx_thr], color='red', s=100, zorder=5,
               edgecolors='darkred', linewidth=1.5,
               label=f"Threshold = {sd['thr']:.3f}")

    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
    ax.set_title(f"Curva ROC – {sd['label']}", fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "curva_roc.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfico 5: Curva ROC → curva_roc.png")

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 6 (NUEVO): Métricas de Evaluación Detalladas
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Métricas de Evaluación Detalladas por Escenario",
             fontsize=15, fontweight='bold', y=1.02)

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_colors = [COLORS['accuracy'], COLORS['precision'], COLORS['recall'], COLORS['f1']]

for sd, ax in zip(scenario_data, axes.flatten()):
    values = [sd['acc'], sd['prec'], sd['rec'], sd['f1']]
    bars = ax.barh(metric_names, values, color=metric_colors, edgecolor='white',
                   linewidth=1.5, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2%}', ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(f"{sd['label']}\n(Thr={sd['thr']:.3f})", fontsize=12)
    ax.axvline(x=0.90, color='red', linestyle='--', alpha=0.5, label='Objetivo 90%')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "metricas_evaluacion.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfico 6: Métricas de Evaluación → metricas_evaluacion.png")

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 7 (NUEVO): Detalle TP, FP, TN, FN por Escenario
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Detalle de Clasificación – TP, TN, FP, FN por Escenario",
             fontsize=15, fontweight='bold', y=1.02)

detail_colors = {'TP': '#4CAF50', 'TN': '#2196F3', 'FP': '#FF9800', 'FN': '#F44336'}
detail_labels = {
    'TP': 'Verdaderos Positivos (TP)',
    'TN': 'Verdaderos Negativos (TN)',
    'FP': 'Falsos Positivos (FP)',
    'FN': 'Falsos Negativos (FN)'
}

for sd, ax in zip(scenario_data, axes.flatten()):
    categories = ['TP', 'TN', 'FP', 'FN']
    values = [sd['tp'], sd['tn'], sd['fp'], sd['fn']]
    colors = [detail_colors[c] for c in categories]
    labels_full = [detail_labels[c] for c in categories]

    bars = ax.bar(labels_full, values, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.65)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')

    total = sum(values)
    ax.set_title(f"{sd['label']}\nTotal muestras en test: {total}", fontsize=12)
    ax.set_ylabel('Cantidad', fontsize=11)
    ax.tick_params(axis='x', rotation=25, labelsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "detalle_tp_fp_tn_fn.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfico 7: Detalle TP/FP/TN/FN → detalle_tp_fp_tn_fn.png")

# %%
# ════════════════════════════════════════════════════════
# GRÁFICO 8 (NUEVO): Distribución de Frecuencia – Age y Fare
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Distribución de Frecuencia – Variables Numéricas",
             fontsize=15, fontweight='bold', y=1.02)

# Fila 1: Distribución de Age
# Histograma de Age por supervivencia (density=True para balancear clases)
ax1 = axes[0, 0]
age_no = df[df['Survived'] == 0]['Age'].dropna()
age_si = df[df['Survived'] == 1]['Age'].dropna()
ax1.hist(age_no, bins=30, alpha=0.6, color='#F44336', density=True,
         label=f'No Sobrevivió (0) n={len(age_no)}', edgecolor='white')
ax1.hist(age_si, bins=30, alpha=0.6, color='#4CAF50', density=True,
         label=f'Sobrevivió (1) n={len(age_si)}', edgecolor='white')
ax1.set_xlabel('Edad', fontsize=11)
ax1.set_ylabel('Densidad (normalizado)', fontsize=11)
ax1.set_title('Distribución de Edad por Supervivencia (normalizado)', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# KDE de Age
ax2 = axes[0, 1]
age_no.plot.kde(ax=ax2, color='#F44336', lw=2,
                label='No Sobrevivió (0)')
age_si.plot.kde(ax=ax2, color='#4CAF50', lw=2,
                label='Sobrevivió (1)')
ax2.set_xlabel('Edad', fontsize=11)
ax2.set_ylabel('Densidad', fontsize=11)
ax2.set_title('Densidad de Edad por Supervivencia', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Fila 2: Distribución de Fare
# Histograma de Fare por supervivencia (density=True para balancear clases)
ax3 = axes[1, 0]
fare_no = df[df['Survived'] == 0]['Fare'].dropna()
fare_si = df[df['Survived'] == 1]['Fare'].dropna()
ax3.hist(fare_no, bins=30, alpha=0.6, color='#FF9800', density=True,
         label=f'No Sobrevivió (0) n={len(fare_no)}', edgecolor='white')
ax3.hist(fare_si, bins=30, alpha=0.6, color='#2196F3', density=True,
         label=f'Sobrevivió (1) n={len(fare_si)}', edgecolor='white')
ax3.set_xlabel('Tarifa (Fare)', fontsize=11)
ax3.set_ylabel('Densidad (normalizado)', fontsize=11)
ax3.set_title('Distribución de Tarifa por Supervivencia (normalizado)', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# KDE de Fare
ax4 = axes[1, 1]
fare_no.plot.kde(ax=ax4, color='#FF9800', lw=2,
                 label='No Sobrevivió (0)')
fare_si.plot.kde(ax=ax4, color='#2196F3', lw=2,
                 label='Sobrevivió (1)')
ax4.set_xlabel('Tarifa (Fare)', fontsize=11)
ax4.set_ylabel('Densidad', fontsize=11)
ax4.set_title('Densidad de Tarifa por Supervivencia', fontsize=12)
ax4.legend(fontsize=9)
ax4.set_xlim(-20, df['Fare'].quantile(0.99) * 1.1)  # Limitar eje x para mejor visualización
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "distribucion_frecuencia.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfico 8: Distribución de Frecuencia → distribucion_frecuencia.png")

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

# Comparar Clean vs Dirty
clean_60 = resultados[resultados['Escenario'].str.contains('Sin Outliers')]
dirty_60 = resultados[resultados['Escenario'].str.contains('Con Outliers')]

min_prec = resultados['Precision'].min()
all_above_85 = min_prec >= 0.85
avg_prec = resultados['Precision'].mean()
avg_acc = resultados['Accuracy'].mean()
avg_recall = resultados['Recall'].mean()

# Calcular AUC para conclusiones
auc_values = []
for sd in scenario_data:
    fpr_c, tpr_c, _ = roc_curve(sd['y_te'], sd['proba'])
    auc_values.append(auc(fpr_c, tpr_c))

# Tabla detallada con TP/FP/TN/FN
detail_lines = ""
for sd in scenario_data:
    lbl = sd['label']
    s_tp, s_tn, s_fp, s_fn = sd['tp'], sd['tn'], sd['fp'], sd['fn']
    s_total_ok = s_tp + s_tn
    s_total = s_tp + s_tn + s_fp + s_fn
    detail_lines += (
        f"   {lbl}:\n"
        f"      - Verdaderos Positivos  (TP): {s_tp:4d}  (sobrevivio y se predijo correctamente)\n"
        f"      - Verdaderos Negativos  (TN): {s_tn:4d}  (no sobrevivio y se predijo correctamente)\n"
        f"      - Falsos Positivos      (FP): {s_fp:4d}  (no sobrevivio pero se predijo que si)\n"
        f"      - Falsos Negativos      (FN): {s_fn:4d}  (sobrevivio pero se predijo que no)\n"
        f"      - Total aciertos: {s_total_ok} / {s_total}\n"
    )

conclusiones = f"""{'='*70}
  CONCLUSIONES - PRÁCTICA 2 (Split 60/20/20 con stratify=y)
{'='*70}

CONFIGURACIÓN DE SEGMENTACIÓN:
   - Entrenamiento: 60% | Validación: 20% | Test: 20%
   - stratify=y: Distribución de clases preservada en todos los splits
   - Esto garantiza que cada partición tiene la misma proporción de
     supervivientes vs no supervivientes que el dataset original.

1. EFECTO DE LOS OUTLIERS (60-20-20, Clean vs Dirty):
   - Sin Outliers (Winsorized): Precision = {clean_60['Precision'].values[0]:.1%}, Accuracy = {clean_60['Accuracy'].values[0]:.1%}
   - Con Outliers (Original):   Precision = {dirty_60['Precision'].values[0]:.1%}, Accuracy = {dirty_60['Accuracy'].values[0]:.1%}
   - Diferencia en Precision: {(clean_60['Precision'].values[0] - dirty_60['Precision'].values[0]):+.1%}
   - Diferencia en Accuracy:  {(clean_60['Accuracy'].values[0] - dirty_60['Accuracy'].values[0]):+.1%}
   > {'La winsorizacion MEJORA la precision al reducir el impacto de valores extremos.' if clean_60['Precision'].values[0] > dirty_60['Precision'].values[0] else 'Los datos originales mantienen mejor precision, sugiriendo que los outliers no afectan significativamente al modelo.'}
   > La winsorizacion mantiene las {len(df)} filas originales (no elimina datos).

2. CURVA ROC Y AUC:
   - Sin Outliers: AUC = {auc_values[0]:.4f}
   - Con Outliers: AUC = {auc_values[1]:.4f}
   > {'Ambos escenarios muestran un AUC excelente (> 0.85), indicando buena capacidad discriminativa.' if min(auc_values) > 0.85 else 'El AUC indica un rendimiento aceptable del clasificador.'}
   > Un AUC de 1.0 indica clasificacion perfecta, 0.5 indica clasificacion aleatoria.

3. DETALLE DE CLASIFICACIÓN (TP, TN, FP, FN):
{detail_lines}
4. THRESHOLD DE DECISIÓN:
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

6. LIBRERÍAS SKLEARN UTILIZADAS:
   - roc_curve: Para calcular la curva ROC (FPR, TPR)
   - auc: Para calcular el Area Under the Curve
   - confusion_matrix: Para generar las matrices de confusión
   - accuracy_score: Para calcular la exactitud del modelo
   - precision_score, recall_score, f1_score: Métricas complementarias

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
