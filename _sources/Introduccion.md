# Predicción de Readmisión Hospitalaria en Pacientes Diabéticos

## Introducción

Este proyecto aborda el problema de **predicción de readmisión hospitalaria en pacientes diabéticos** utilizando técnicas de Machine Learning. El objetivo principal es desarrollar un modelo predictivo capaz de identificar pacientes con alto riesgo de ser readmitidos en un plazo de 30 días o más tras su alta hospitalaria, permitiendo implementar intervenciones preventivas tempranas.

El dataset utilizado contiene **101,766 registros** de encuentros hospitalarios de pacientes diabéticos, con 50 características que incluyen información demográfica, diagnósticos, medicación, procedimientos de laboratorio y resultados clínicos. La variable objetivo (`target`) clasifica a los pacientes en dos categorías: readmitidos (47%) y no readmitidos (53%), presentando un desbalance moderado que fue abordado mediante técnicas de balanceo.

A lo largo del proyecto se implementaron **10 modelos de clasificación** diferentes, desde algoritmos clásicos como KNN y Naive Bayes hasta modelos de ensamble como Random Forest y XGBoost. El modelo final seleccionado fue **XGBoost optimizado** con técnicas avanzadas de tuning (Optuna), logrando un **F1-score de 0.5682** y un **AUC de 0.6787** en el conjunto de test.

## Contexto

La readmisión hospitalaria de pacientes diabéticos representa un problema crítico en el sistema de salud, tanto por el impacto en la calidad de vida del paciente como por los costos económicos asociados. Identificar de manera anticipada qué pacientes tienen mayor probabilidad de reingreso permite:

- **Optimizar recursos médicos:** Enfocando intervenciones preventivas en pacientes de alto riesgo.
- **Mejorar resultados clínicos:** Implementando planes de seguimiento personalizados.
- **Reducir costos:** Previniendo hospitalizaciones innecesarias.

El proyecto se motivó por la necesidad de desarrollar una herramienta de apoyo a la toma de decisiones clínicas basada en datos históricos. Utilizando el **Dataset de Diabetes 130-US hospitals (1999-2008)**, disponible en el UCI Machine Learning Repository, se exploraron patrones en variables demográficas (raza, género, edad), clínicas (diagnósticos, medicamentos, procedimientos de laboratorio) y operacionales (tipo de admisión, tiempo de hospitalización).

Los aspectos específicos explorados incluyen:
- **Análisis exploratorio exhaustivo (EDA):** Identificación de valores faltantes (>97% en `weight`), detección de outliers, análisis de correlaciones y reducción de dimensionalidad mediante PCA y t-SNE.
- **Técnicas de balanceo:** Comparación de SMOTE, class weighting y ADASYN para manejar el desbalance de clases.
- **Optimización de modelos:** Evaluación de técnicas especializadas como Ball Trees para KNN, solver 'saga' para modelos lineales, y `tree_method='hist'` para XGBoost.
- **Interpretabilidad:** Aplicación de LIME y SHAP para explicar las predicciones del modelo final.

## Índice

### 1. [Análisis Exploratorio de Datos (EDA)](notebooks/1_EDA.ipynb)
   - Descripción del dataset y variables
   - Análisis de balance de clases
   - Identificación y tratamiento de valores faltantes
   - Detección de outliers mediante método IQR
   - Matriz de correlaciones y análisis de colinealidad
   - Reducción de dimensionalidad (PCA y t-SNE)
   - Análisis univariado por variables clave (raza, género, edad, medicamentos)

### 2. [Preprocesamiento de Datos](notebooks/1_5_Preprocesamiento.ipynb)
   - Limpieza y eliminación de columnas irrelevantes
   - Agrupación de categorías (admission_type, admission_source)
   - Manejo de valores faltantes con estrategias de imputación
   - Codificación de variables categóricas (One-Hot Encoding)
   - Codificación ordinal de medicamentos (No, Steady, Down, Up)
   - Escalado de variables numéricas (StandardScaler)
   - División estratificada (Train 70% / Validation 15% / Test 15%)
   - Aplicación de SMOTE para balanceo de clases en Train

### 3. [Benchmark de Modelos Base](notebooks/2_Benchmark.ipynb)
   - Evaluación de 10 modelos de clasificación:
     * K-Nearest Neighbors (KNN)
     * Naive Bayes
     * Regresión Logística (L1 y L2)
     * Ridge Classifier
     * Lasso (SGD)
     * Árbol de Decisión
     * Random Forest
     * XGBoost
     * SVM Lineal
   - Métricas de evaluación: Accuracy, F1-score, AUC-ROC, Tiempo de entrenamiento
   - Análisis detallado del modelo XGBoost con LIME
   - Comparación visual de resultados

### 4. [Técnicas de Balanceo de Clases](notebooks/3_Balanceo.ipynb)
   - Evaluación de estrategias de balanceo:
     * SMOTE (Synthetic Minority Over-sampling Technique)
     * Class Weight Balancing
     * Comparación con ADASYN (no aplicable por distribución equilibrada)
   - Análisis de impacto en métricas de clasificación
   - Selección de estrategia óptima por modelo

### 5. [Optimización de Modelos](notebooks/4_Optimizacion.ipynb)
   - Técnicas de optimización especializadas por modelo:
     * KNN: KD-Trees, Ball Trees, Brute Force
     * Naive Bayes: `partial_fit()` con entrenamiento por lotes
     * Ridge/Lasso: Solver 'saga' optimizado
     * XGBoost: `tree_method='hist'` + early stopping
     * SVM: LinearSVC vs SGDClassifier
   - Métricas de optimización: Tiempo de entrenamiento, uso de memoria, velocidad de predicción
   - Análisis comparativo de mejoras por modelo

### 6. [Evaluación Completa de Modelos](notebooks/5_Evaluacion.ipynb)
   - Evaluación exhaustiva de los 10 modelos optimizados
   - Matrices de confusión y análisis de errores
   - Curvas ROC y cálculo de AUC
   - Métricas detalladas: Accuracy, Precision, Recall, F1-score
   - Justificación de F1-score como métrica principal
   - Interpretación clínica de resultados
   - Identificación del mejor modelo: **XGBoost (F1=0.5682, AUC=0.6787)**

### 7. [Validación Avanzada y Tuning](notebooks/6_Validacion_Tuning.ipynb)
   - Validación cruzada estratificada (StratifiedKFold, 5 folds)
   - Optimización de hiperparámetros:
     * XGBoost: OptunaSearchCV (80 trials)
     * Árbol de Decisión: GridSearchCV
   - Interpretabilidad avanzada:
     * Feature Importances
     * SHAP (Shapley Additive Explanations)
     * Permutation Importance
   - Análisis de las variables más influyentes
   - Comparación final en conjuntos de Validación y Test

## Resultados Principales

- **Modelo Final:** XGBoost optimizado con Optuna
- **Métricas en Test:**
  - F1-score: **0.5682**
  - Precision: **0.6229**
  - Recall: **0.5224**
  - AUC: **0.6787**
- **Variables más influyentes:** number_inpatient, number_diagnoses, time_in_hospital, discharge_disposition_id
- **Técnica de balanceo seleccionada:** SMOTE con `sampling_strategy='auto'`
- **Optimización lograda:** Reducción del 50-60% en tiempo de entrenamiento sin pérdida de rendimiento

## Participantes

- **José Manuel Menco Galván**
- **Camilo Andrés Vargas Escorcia**
- **Iván Alejandro Ramírez Mejía**

---

## Tecnologías Utilizadas

- **Python 3.x**
- **Bibliotecas principales:**
  - Pandas, NumPy (manipulación de datos)
  - Scikit-learn (modelado y preprocesamiento)
  - XGBoost (gradient boosting optimizado)
  - Imbalanced-learn (técnicas de balanceo)
  - Optuna (optimización bayesiana de hiperparámetros)
  - SHAP, LIME (interpretabilidad)
  - Matplotlib, Seaborn (visualización)

## Estructura del Proyecto

```
MachineLearning/
├── data/
│   ├── raw/                      # Dataset original
│   ├── processed/                # Datos preprocesados (Train/Val/Test)
│   └── transformers/             # Scalers y encoders guardados
├── notebooks/
│   ├── 1_EDA.ipynb              # Análisis exploratorio
│   ├── 1_5_Preprocesamiento.ipynb
│   ├── 2_Benchmark.ipynb
│   ├── 3_Balanceo.ipynb
│   ├── 4_Optimizacion.ipynb
│   ├── 5_Evaluacion.ipynb
│   └── 6_Validacion_Tuning.ipynb
└── Introduccion.md               # Este documento
```