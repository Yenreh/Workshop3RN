"""
preprocessing.py
Módulo para normalización de datos y creación de ventanas temporales.
"""

import numpy as np
from tensorflow import keras


class TimeSeriesNormalizer:
    """
    Clase para normalizar series temporales usando z-score.
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data):
        """
        Calcula la media y desviación estándar de los datos de entrenamiento.
        
        Args:
            data (np.ndarray): Datos de entrenamiento
        """
        self.mean = np.mean(data)
        self.std = np.std(data)
        
    def transform(self, data):
        """
        Normaliza los datos usando la media y std calculadas.
        
        Args:
            data (np.ndarray): Datos a normalizar
            
        Returns:
            np.ndarray: Datos normalizados
        """
        if self.mean is None or self.std is None:
            raise ValueError("El normalizador no ha sido ajustado. Ejecute fit() primero.")
        
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        """
        Ajusta y transforma los datos en un solo paso.
        
        Args:
            data (np.ndarray): Datos a ajustar y normalizar
            
        Returns:
            np.ndarray: Datos normalizados
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        """
        Desnormaliza los datos.
        
        Args:
            data (np.ndarray): Datos normalizados
            
        Returns:
            np.ndarray: Datos en escala original
        """
        if self.mean is None or self.std is None:
            raise ValueError("El normalizador no ha sido ajustado.")
        
        return data * self.std + self.mean


def create_sequences(data, sequence_length, prediction_horizon=1):
    """
    Crea secuencias de ventanas temporales para entrenamiento.
    
    Args:
        data (np.ndarray): Serie temporal de datos
        sequence_length (int): Longitud de la ventana temporal (número de pasos pasados)
        prediction_horizon (int): Horizonte de predicción (pasos hacia el futuro)
        
    Returns:
        tuple: (X, y) donde X son las secuencias de entrada y y los objetivos
    """
    X = []
    y = []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Secuencia de entrada
        sequence = data[i:i + sequence_length]
        # Objetivo (valor futuro)
        target = data[i + sequence_length + prediction_horizon - 1]
        
        X.append(sequence)
        y.append(target)
    
    return np.array(X), np.array(y)


def create_sequences_multivariate(data, sequence_length, prediction_horizon=1):
    """
    Crea secuencias para datos multivariados.
    
    Args:
        data (np.ndarray): Serie temporal multivariada (shape: [n_samples, n_features])
        sequence_length (int): Longitud de la ventana temporal
        prediction_horizon (int): Horizonte de predicción
        
    Returns:
        tuple: (X, y) donde X son las secuencias de entrada y y los objetivos
    """
    X = []
    y = []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Secuencia de entrada
        sequence = data[i:i + sequence_length]
        # Objetivo (asumiendo que queremos predecir la primera característica)
        target = data[i + sequence_length + prediction_horizon - 1, 0]
        
        X.append(sequence)
        y.append(target)
    
    return np.array(X), np.array(y)


def prepare_sequences_for_training(train_data, val_data, test_data, 
                                   seq_length=60, prediction_horizon=1,
                                   normalize=True):
    """
    Prepara secuencias normalizadas para entrenamiento, validación y prueba.
    
    Args:
        train_data (np.ndarray): Datos de entrenamiento
        val_data (np.ndarray): Datos de validación
        test_data (np.ndarray): Datos de prueba
        seq_length (int): Longitud de la ventana temporal
        prediction_horizon (int): Horizonte de predicción (por defecto 1 = 24h)
        normalize (bool): Si True, normaliza los datos
        
    Returns:
        dict: Diccionario con claves:
            - X_train, y_train: Datos de entrenamiento
            - X_val, y_val: Datos de validación
            - X_test, y_test: Datos de prueba
            - normalizer: Normalizador ajustado
    """
    normalizer = None
    
    if normalize:
        # Normalizar usando solo los datos de entrenamiento
        normalizer = TimeSeriesNormalizer()
        train_data_norm = normalizer.fit_transform(train_data)
        val_data_norm = normalizer.transform(val_data)
        test_data_norm = normalizer.transform(test_data)
    else:
        train_data_norm = train_data
        val_data_norm = val_data
        test_data_norm = test_data
    
    # Crear secuencias
    X_train, y_train = create_sequences(train_data_norm, seq_length, prediction_horizon)
    X_val, y_val = create_sequences(val_data_norm, seq_length, prediction_horizon)
    X_test, y_test = create_sequences(test_data_norm, seq_length, prediction_horizon)
    
    # Reshape para LSTM/GRU: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'normalizer': normalizer
    }


def create_tf_dataset(X, y, batch_size=32, shuffle=True):
    """
    Crea un TensorFlow Dataset para entrenamiento eficiente.
    
    Args:
        X (np.ndarray): Datos de entrada
        y (np.ndarray): Objetivos
        batch_size (int): Tamaño del batch
        shuffle (bool): Si True, mezcla los datos
        
    Returns:
        tf.data.Dataset: Dataset de TensorFlow
    """
    dataset = keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=y,
        sequence_length=None,  # Ya tenemos las secuencias creadas
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataset


def print_sequences_info(X_train, y_train, X_val, y_val, X_test, y_test, 
                        sequence_length, prediction_horizon=1):
    """
    Imprime información sobre las secuencias creadas.
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Conjuntos de datos
        sequence_length (int): Longitud de la secuencia
        prediction_horizon (int): Horizonte de predicción
    """
    print(f"\n{'='*60}")
    print(f"Información de Secuencias Temporales")
    print(f"{'='*60}")
    print(f"Longitud de secuencia: {sequence_length} días")
    print(f"Horizonte de predicción: {prediction_horizon} día(s) adelante")
    print(f"\nForma de los datos:")
    print(f"  X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape} | y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape} | y_test:  {y_test.shape}")
    print(f"\nInterpretación:")
    print(f"  Cada muestra contiene {sequence_length} días pasados")
    print(f"  para predecir el valor {prediction_horizon} día(s) en el futuro")
    print(f"{'='*60}\n")


def get_sequence_statistics(data, sequence_length):
    """
    Calcula estadísticas sobre las secuencias que se pueden crear.
    
    Args:
        data (np.ndarray): Datos originales
        sequence_length (int): Longitud de la secuencia
        
    Returns:
        dict: Diccionario con estadísticas
    """
    total_samples = len(data)
    possible_sequences = max(0, total_samples - sequence_length)
    
    stats = {
        'total_samples': total_samples,
        'sequence_length': sequence_length,
        'possible_sequences': possible_sequences,
        'data_coverage': possible_sequences / total_samples if total_samples > 0 else 0
    }
    
    return stats
