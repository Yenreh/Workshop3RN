"""
train_gru.py
Script para entrenar modelos GRU con diferentes configuraciones.
"""

import argparse
import os
import sys
import numpy as np
from data_loader import load_and_prepare_data
from preprocessing import prepare_sequences_for_training, print_sequences_info
from models import create_gru_model, print_model_summary
from utils import create_callbacks, evaluate_model, save_model_info
from visualize import plot_training_history, plot_predictions, plot_residuals


def train_gru(stock_file, seq_length, epochs, batch_size, units, 
             recurrent_dropout, learning_rate, output_dir='output',
             train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Entrena un modelo GRU para predicción de precios de acciones.
    
    Args:
        stock_file (str): Ruta al archivo de datos de la acción
        seq_length (int): Longitud de la secuencia temporal
        epochs (int): Número de épocas de entrenamiento
        batch_size (int): Tamaño del batch
        units (int): Número de unidades GRU
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
    print("ENTRENAMIENTO DE MODELO GRU")
    print("="*70)
    
    # Configuración del experimento
    config = {
        'model_type': 'gru',
        'stock_file': stock_file,
        'sequence_length': seq_length,
        'epochs': epochs,
        'batch_size': batch_size,
        'units': units,
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
    print("\nCreando modelo GRU...")
    model = create_gru_model(
        input_shape=(seq_length, 1),
        units=units,
        recurrent_dropout=recurrent_dropout
    )
    model.summary()
    
    # 4. Preparar callbacks
    model_name = f"gru_seq{seq_length}_batch{batch_size}_dropout{int(recurrent_dropout*100)}"
    callbacks = create_callbacks(model_name, output_dir=output_dir, patience=10)
    
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
    
    # Historial de entrenamiento
    plot_training_history(
        history, 
        model_name=model_name,
        save_path=os.path.join(output_dir, f"{model_name}_history.png")
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
        save_path=os.path.join(output_dir, f"{model_name}_predictions.png")
    )
    
    # Residuos
    plot_residuals(
        y_test_original,
        y_pred_original,
        model_name=model_name,
        save_path=os.path.join(output_dir, f"{model_name}_residuals.png")
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


def main():
    """Función principal para ejecutar desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo GRU para predicción de precios de acciones'
    )
    
    # Argumentos requeridos
    parser.add_argument('--stock_file', type=str, required=True,
                       help='Ruta al archivo de datos de la acción')
    
    # Hiperparámetros del modelo
    parser.add_argument('--seq_length', type=int, default=120,
                       help='Longitud de la secuencia temporal (default: 120)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas de entrenamiento (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño del batch (default: 32)')
    parser.add_argument('--units', type=int, default=64,
                       help='Número de unidades GRU (default: 64)')
    parser.add_argument('--recurrent_dropout', type=float, default=0.0,
                       help='Tasa de dropout recurrente (default: 0.0)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Tasa de aprendizaje (default: 0.001)')
    
    # División de datos
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proporción de datos de entrenamiento (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Proporción de datos de validación (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Proporción de datos de prueba (default: 0.15)')
    
    # Directorio de salida
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directorio para guardar resultados (default: output)')
    
    args = parser.parse_args()
    
    # Verificar que el archivo existe
    if not os.path.exists(args.stock_file):
        print(f"❌ Error: No se encontró el archivo {args.stock_file}")
        sys.exit(1)
    
    # Verificar proporciones
    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        print("❌ Error: Las proporciones de división deben sumar 1.0")
        sys.exit(1)
    
    # Entrenar modelo
    try:
        results = train_gru(
            stock_file=args.stock_file,
            seq_length=args.seq_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            units=args.units,
            recurrent_dropout=args.recurrent_dropout,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        print(f"\n🎉 Modelo '{results['model_name']}' entrenado exitosamente!")
        print(f"📁 Resultados guardados en: {args.output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
