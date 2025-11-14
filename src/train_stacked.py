"""
train_stacked.py
Script para entrenar modelos recurrentes apilados (LSTM o GRU).
"""

import argparse
import os
import sys
import numpy as np
from data_loader import load_and_prepare_data
from preprocessing import prepare_sequences_for_training, print_sequences_info
from models import (create_stacked_lstm_model, create_stacked_gru_model, 
                   print_model_summary)
from utils import create_callbacks, evaluate_model, save_model_info
from visualize import plot_training_history, plot_predictions, plot_residuals


def train_stacked(stock_file, model_type, seq_length, epochs, batch_size, 
                 units_list, recurrent_dropout, learning_rate, 
                 output_dir='output', train_ratio=0.7, val_ratio=0.15, 
                 test_ratio=0.15):
    """
    Entrena un modelo recurrente apilado para predicción de precios.
    
    Args:
        stock_file (str): Ruta al archivo de datos de la acción
        model_type (str): Tipo de capa recurrente ('lstm' o 'gru')
        seq_length (int): Longitud de la secuencia temporal
        epochs (int): Número de épocas de entrenamiento
        batch_size (int): Tamaño del batch
        units_list (list): Lista con número de unidades en cada capa
        recurrent_dropout (float): Tasa de dropout recurrente
        learning_rate (float): Tasa de aprendizaje
        output_dir (str): Directorio para guardar resultados
        train_ratio (float): Proporción de entrenamiento
        val_ratio (float): Proporción de validación
        test_ratio (float): Proporción de prueba
    
    Returns:
        dict: Diccionario con resultados del entrenamiento
    """
    print("\n" + "="*70)
    print(f"ENTRENAMIENTO DE MODELO {model_type.upper()} APILADO")
    print("="*70)
    
    # Configuración del experimento
    config = {
        'model_type': f'stacked_{model_type}',
        'stock_file': stock_file,
        'sequence_length': seq_length,
        'epochs': epochs,
        'batch_size': batch_size,
        'units_list': units_list,
        'num_layers': len(units_list),
        'recurrent_dropout': recurrent_dropout,
        'learning_rate': learning_rate,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }
    
    print("Configuración:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 1. Cargar datos
    print("\nCargando datos...")
    data_dict = load_and_prepare_data(
        stock_file, train_ratio, val_ratio, test_ratio
    )
    train_data = data_dict['train_data']
    val_data = data_dict['val_data']
    test_data = data_dict['test_data']
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # 2. Preparar secuencias
    print("\nPreparando secuencias temporales...")
    sequences = prepare_sequences_for_training(
        train_data, val_data, test_data, seq_length
    )
    
    X_train = sequences['X_train']
    y_train = sequences['y_train']
    X_val = sequences['X_val']
    y_val = sequences['y_val']
    X_test = sequences['X_test']
    y_test = sequences['y_test']
    normalizer = sequences['normalizer']
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # 3. Crear modelo
    print(f"\nCreando modelo {model_type.upper()} apilado...")
    print(f"  Número de capas: {len(units_list)}")
    print(f"  Unidades por capa: {units_list}")
    
    if model_type.lower() == 'lstm':
        model = create_stacked_lstm_model(
            input_shape=(seq_length, 1),
            units_list=units_list,
            recurrent_dropout=recurrent_dropout
        )
    elif model_type.lower() == 'gru':
        model = create_stacked_gru_model(
            input_shape=(seq_length, 1),
            units_list=units_list,
            recurrent_dropout=recurrent_dropout
        )
    else:
        raise ValueError(f"Tipo de modelo no válido: {model_type}. Use 'lstm' o 'gru'.")
    
    model.summary()
    
    # 4. Preparar callbacks
    layers_str = f"layers{len(units_list)}"
    dropout_str = f"dropout{int(recurrent_dropout*100)}"
    model_name = f"stacked_{model_type}_seq{seq_length}_batch{batch_size}_{layers_str}_{dropout_str}"
    
    callbacks = create_callbacks(model_name, output_dir=output_dir, patience=15)
    
    # 5. Entrenar modelo
    print(f"\nIniciando entrenamiento (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nEntrenamiento completado!")
    
    # 6. Evaluar modelo
    print("\nEvaluando modelo en conjunto de prueba...")
    metrics = evaluate_model(model, X_test, y_test, normalizer)
    
    print("\nMétricas:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # 7. Guardar resultados
    print("\nGuardando resultados...")
    save_model_info(model, config, metrics, model_name, output_dir)
    
    # 8. Visualizar resultados
    print("\nGenerando visualizaciones...")
    
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Historial de entrenamiento
    plot_training_history(
        history, 
        model_name=model_name,
        save_path=os.path.join(images_dir, f"{model_name}_history.png")
    )
    
    # Predicciones
    y_pred = model.predict(X_test, verbose=0)
    y_test_original = normalizer.inverse_transform(y_test)
    y_pred_original = normalizer.inverse_transform(y_pred)
    
    plot_predictions(
        y_test_original,
        y_pred_original,
        model_name=model_name,
        n_points=200,
        save_path=os.path.join(images_dir, f"{model_name}_predictions.png")
    )
    
    # Residuos
    plot_residuals(
        y_test_original,
        y_pred_original,
        model_name=model_name,
        save_path=os.path.join(images_dir, f"{model_name}_residuals.png")
    )
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    
    return {
        'model_name': model_name,
        'model': model,
        'history': history,
        'metrics': metrics,
        'config': config
    }
