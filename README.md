# 🧪 MLOps Workshop - PyCon 2023: Logistic Regression with Iris Dataset

Este repositorio presenta una implementación práctica de un flujo de trabajo MLOps utilizando [Weights & Biases (WandB)](https://wandb.ai/) para la gestión del ciclo de vida del modelo de Machine Learning y [GitHub Actions](https://github.com/features/actions) para la integración y entrega continua. El modelo aplicado es una **regresión logística multinomial** entrenada con el famoso dataset **Iris**.

## 🚀 Objetivos del Proyecto

- Implementar un pipeline reproducible para el entrenamiento y despliegue de modelos.
- Usar WandB para el tracking de experimentos y gestión de artefactos.
- Automatizar el flujo de trabajo ML con GitHub Actions.
- Aplicar buenas prácticas de MLOps desde la preparación de datos hasta el monitoreo del modelo.
- Resolver un problema de clasificación multiclase utilizando un modelo de regresión logística.

## 🎯 Problema de Predicción

El objetivo de este proyecto es **predecir la especie de una flor** a partir de sus características físicas:
- Largo y ancho del sépalo.
- Largo y ancho del pétalo.

El modelo se entrena para **clasificar correctamente** cada instancia en una de las tres especies de iris:
- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

Se utiliza una **regresión logística multinomial**, adecuada para este tipo de problema de clasificación multiclase, buscando maximizar la precisión general y minimizar los errores de predicción en cada categoría.

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.9+
- **Machine Learning:** Scikit-learn (Regresión logística multinomial)
- **Tracking de experimentos:** Weights & Biases (WandB)
- **Automatización de flujos:** GitHub Actions
- **Serialización de modelos:** `joblib`

## 🔄 Flujo de Trabajo del Proyecto

```mermaid
flowchart TD
    A[Raw Data (Iris CSV)] --> B[Preprocesamiento (normalización)]
    B --> C[División de datos (train/validation/test)]
    C --> D[Entrenamiento del modelo (Regresión logística)]
    D --> E[Evaluación del modelo (métricas, confusion matrix)]
    E --> F[Registro de artefactos y métricas en WandB]
