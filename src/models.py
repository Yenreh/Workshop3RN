"""
models.py
Módulo con definiciones de modelos LSTM, GRU y redes apiladas.
"""

from tensorflow import keras
from tensorflow.keras import layers


def create_lstm_model(input_shape=None, sequence_length=None, units=64, 
                      recurrent_dropout=0.0, learning_rate=0.001):
    """
    Crea un modelo LSTM para predicción de series temporales.
    
    Args:
        input_shape (tuple): Forma de la entrada (seq_length, features). Si se proporciona, se usa este.
        sequence_length (int): Longitud de la secuencia (solo si input_shape es None)
        units (int): Número de unidades en la capa LSTM
        recurrent_dropout (float): Tasa de dropout recurrente
        learning_rate (float): Tasa de aprendizaje del optimizador
        
    Returns:
        keras.Model: Modelo LSTM compilado
    """
    # Determinar la forma de entrada
    if input_shape is not None:
        shape = input_shape
    elif sequence_length is not None:
        shape = (sequence_length, 1)
    else:
        raise ValueError("Debe proporcionar input_shape o sequence_length")
    
    model = keras.Sequential([
        layers.Input(shape=shape),
        layers.LSTM(units, recurrent_dropout=recurrent_dropout),
        layers.Dense(1)
    ], name="LSTM_Model")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_gru_model(input_shape=None, sequence_length=None, units=64, 
                     recurrent_dropout=0.0, learning_rate=0.001):
    """
    Crea un modelo GRU para predicción de series temporales.
    
    Args:
        input_shape (tuple): Forma de la entrada (seq_length, features). Si se proporciona, se usa este.
        sequence_length (int): Longitud de la secuencia (solo si input_shape es None)
        units (int): Número de unidades en la capa GRU
        recurrent_dropout (float): Tasa de dropout recurrente
        learning_rate (float): Tasa de aprendizaje del optimizador
        
    Returns:
        keras.Model: Modelo GRU compilado
    """
    # Determinar la forma de entrada
    if input_shape is not None:
        shape = input_shape
    elif sequence_length is not None:
        shape = (sequence_length, 1)
    else:
        raise ValueError("Debe proporcionar input_shape o sequence_length")
    
    model = keras.Sequential([
        layers.Input(shape=shape),
        layers.GRU(units, recurrent_dropout=recurrent_dropout),
        layers.Dense(1)
    ], name="GRU_Model")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_stacked_lstm_model(input_shape=None, sequence_length=None, 
                              units_list=[64, 32], recurrent_dropout=0.0, 
                              learning_rate=0.001):
    """
    Crea un modelo LSTM con capas apiladas.
    
    Args:
        input_shape (tuple): Forma de la entrada (seq_length, features). Si se proporciona, se usa este.
        sequence_length (int): Longitud de la secuencia (solo si input_shape es None)
        units_list (list): Lista con número de unidades en cada capa LSTM
        recurrent_dropout (float): Tasa de dropout recurrente
        learning_rate (float): Tasa de aprendizaje del optimizador
        
    Returns:
        keras.Model: Modelo LSTM apilado compilado
    """
    # Determinar la forma de entrada
    if input_shape is not None:
        shape = input_shape
    elif sequence_length is not None:
        shape = (sequence_length, 1)
    else:
        raise ValueError("Debe proporcionar input_shape o sequence_length")
    
    layers_list = [layers.Input(shape=shape)]
    
    # Añadir capas LSTM
    for i, units in enumerate(units_list):
        # Todas las capas excepto la última deben retornar secuencias
        return_sequences = (i < len(units_list) - 1)
        layers_list.append(
            layers.LSTM(
                units, 
                return_sequences=return_sequences,
                recurrent_dropout=recurrent_dropout
            )
        )
    
    # Capa de salida
    layers_list.append(layers.Dense(1))
    
    model = keras.Sequential(layers_list, name="Stacked_LSTM_Model")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_stacked_gru_model(input_shape=None, sequence_length=None, 
                             units_list=[64, 32], recurrent_dropout=0.0, 
                             learning_rate=0.001):
    """
    Crea un modelo GRU con capas apiladas.
    
    Args:
        input_shape (tuple): Forma de la entrada (seq_length, features). Si se proporciona, se usa este.
        sequence_length (int): Longitud de la secuencia (solo si input_shape es None)
        units_list (list): Lista con número de unidades en cada capa GRU
        recurrent_dropout (float): Tasa de dropout recurrente
        learning_rate (float): Tasa de aprendizaje del optimizador
        
    Returns:
        keras.Model: Modelo GRU apilado compilado
    """
    # Determinar la forma de entrada
    if input_shape is not None:
        shape = input_shape
    elif sequence_length is not None:
        shape = (sequence_length, 1)
    else:
        raise ValueError("Debe proporcionar input_shape o sequence_length")
    
    layers_list = [layers.Input(shape=shape)]
    
    # Añadir capas GRU
    for i, units in enumerate(units_list):
        # Todas las capas excepto la última deben retornar secuencias
        return_sequences = (i < len(units_list) - 1)
        layers_list.append(
            layers.GRU(
                units,
                return_sequences=return_sequences,
                recurrent_dropout=recurrent_dropout
            )
        )
    
    # Capa de salida
    layers_list.append(layers.Dense(1))
    
    model = keras.Sequential(layers_list, name="Stacked_GRU_Model")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_bidirectional_lstm_model(sequence_length, units=64, 
                                   recurrent_dropout=0.0, learning_rate=0.001):
    """
    Crea un modelo LSTM bidireccional.
    
    Args:
        sequence_length (int): Longitud de la secuencia de entrada
        units (int): Número de unidades en la capa LSTM
        recurrent_dropout (float): Tasa de dropout recurrente
        learning_rate (float): Tasa de aprendizaje del optimizador
        
    Returns:
        keras.Model: Modelo LSTM bidireccional compilado
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, 1)),
        layers.Bidirectional(
            layers.LSTM(units, recurrent_dropout=recurrent_dropout)
        ),
        layers.Dense(1)
    ], name="Bidirectional_LSTM_Model")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model


def print_model_summary(model):
    """
    Imprime un resumen del modelo.
    
    Args:
        model (keras.Model): Modelo a resumir
    """
    print(f"\n{'='*60}")
    print(f"Resumen del Modelo: {model.name}")
    print(f"{'='*60}")
    model.summary()
    
    # Contar parámetros
    trainable_params = sum([layer.count_params() for layer in model.trainable_weights])
    non_trainable_params = sum([layer.count_params() for layer in model.non_trainable_weights])
    
    print(f"\nParámetros entrenables: {trainable_params:,}")
    print(f"Parámetros no entrenables: {non_trainable_params:,}")
    print(f"Total de parámetros: {trainable_params + non_trainable_params:,}")
    print(f"{'='*60}\n")
