"""
utils.py
Funciones auxiliares para entrenamiento y evaluación de modelos.
"""

import numpy as np
import os
import pandas as pd
from tensorflow import keras


def create_callbacks(model_name, output_dir='output', patience=10):
    """
    Crea callbacks para el entrenamiento.
    
    Args:
        model_name (str): Nombre del modelo
        output_dir (str): Directorio para guardar modelos
        patience (int): Paciencia para early stopping
        
    Returns:
        list: Lista de callbacks
    """
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{model_name}_best.keras")
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience // 2,
        mode='min',
        min_lr=1e-7,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]


def evaluate_model(model, X_test, y_test, normalizer=None):
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Objetivos de prueba
        normalizer: Normalizador (opcional)
        
    Returns:
        dict: Métricas de evaluación
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    if normalizer is not None:
        y_test_original = normalizer.inverse_transform(y_test)
        y_pred_original = normalizer.inverse_transform(y_pred)
    else:
        y_test_original = y_test
        y_pred_original = y_pred
    
    mse = np.mean((y_test_original - y_pred_original) ** 2)
    mae = np.mean(np.abs(y_test_original - y_pred_original))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_test_original - y_pred_original) ** 2)
    ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }
    
    return metrics


def save_model_info(model, config, metrics, model_name, output_dir='output'):
    """
    Guarda información del modelo en formato CSV.
    
    Args:
        model: Modelo entrenado
        config: Configuración del experimento
        metrics: Métricas de evaluación
        model_name: Nombre del modelo
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'results.csv')
    
    data = {
        'model_name': model_name,
        **config,
        **metrics
    }
    
    df = pd.DataFrame([data])
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    print(f"Resultados guardados en: {csv_path}")


def compare_models(results_list):
    """
    Compara múltiples modelos.
    
    Args:
        results_list: Lista de diccionarios con resultados
        
    Returns:
        DataFrame con comparación
    """
    import pandas as pd
    
    comparison_data = []
    for result in results_list:
        comparison_data.append({
            'Modelo': result['config']['name'],
            'MSE': result['metrics']['mse'],
            'MAE': result['metrics']['mae'],
            'RMSE': result['metrics']['rmse'],
            'R²': result['metrics']['r2'],
            'MAPE (%)': result['metrics']['mape']
        })
    
    df = pd.DataFrame(comparison_data)
    return df
