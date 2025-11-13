"""
data_loader.py
Módulo para cargar y procesar datos de acciones del mercado financiero.
"""

import pandas as pd
import numpy as np
import os


def load_stock_data(file_path):
    """
    Carga datos de una acción desde un archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo de datos de la acción
        
    Returns:
        pd.DataFrame: DataFrame con los datos de la acción
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no tiene el formato esperado
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")
    
    try:
        # Cargar datos
        df = pd.read_csv(file_path)
        
        # Verificar columnas requeridas
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Convertir la columna Date a datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ordenar por fecha
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {str(e)}")


def calculate_average_price(df):
    """
    Calcula el precio promedio (High + Low) / 2 para cada día.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de la acción
        
    Returns:
        pd.DataFrame: DataFrame con columna adicional 'Average_Price'
    """
    df = df.copy()
    df['Average_Price'] = (df['High'] + df['Low']) / 2
    return df


def get_average_price_series(df):
    """
    Extrae la serie temporal del precio promedio.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de la acción
        
    Returns:
        np.ndarray: Array con los precios promedios
    """
    if 'Average_Price' not in df.columns:
        df = calculate_average_price(df)
    
    return df['Average_Price'].values


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        data (np.ndarray): Datos a dividir
        train_ratio (float): Proporción de datos para entrenamiento
        val_ratio (float): Proporción de datos para validación
        test_ratio (float): Proporción de datos para prueba
        
    Returns:
        tuple: (train_data, val_data, test_data, indices)
            donde indices es un dict con los índices de cada conjunto
            
    Raises:
        ValueError: Si las proporciones no suman 1
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Las proporciones deben sumar 1")
    
    n_samples = len(data)
    
    # Calcular índices de división
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    # Dividir datos
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Guardar índices para referencia
    indices = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, n_samples)
    }
    
    return train_data, val_data, test_data, indices


def print_data_info(df, stock_name=""):
    """
    Imprime información resumida del dataset.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        stock_name (str): Nombre de la acción
    """
    print(f"\n{'='*60}")
    print(f"Información del Dataset{': ' + stock_name if stock_name else ''}")
    print(f"{'='*60}")
    print(f"Número total de registros: {len(df)}")
    print(f"Rango de fechas: {df['Date'].min()} a {df['Date'].max()}")
    print(f"Número de días: {(df['Date'].max() - df['Date'].min()).days}")
    print(f"\nEstadísticas del precio promedio:")
    
    if 'Average_Price' in df.columns:
        avg_price = df['Average_Price']
        print(f"  Mínimo: ${avg_price.min():.2f}")
        print(f"  Máximo: ${avg_price.max():.2f}")
        print(f"  Promedio: ${avg_price.mean():.2f}")
        print(f"  Desviación estándar: ${avg_price.std():.2f}")
    
    print(f"{'='*60}\n")


def load_and_prepare_data(file_path, train_ratio=0.7, val_ratio=0.15, 
                          test_ratio=0.15, verbose=True):
    """
    Función de alto nivel para cargar y preparar datos.
    
    Args:
        file_path (str): Ruta al archivo de datos
        train_ratio (float): Proporción de entrenamiento
        val_ratio (float): Proporción de validación
        test_ratio (float): Proporción de prueba
        verbose (bool): Si True, imprime información
        
    Returns:
        dict: Diccionario con claves:
            - df: DataFrame original
            - average_prices: Serie de precios promedio
            - train_data: Datos de entrenamiento
            - val_data: Datos de validación
            - test_data: Datos de prueba
            - train_idx: Índices de entrenamiento
            - val_idx: Índices de validación
            - test_idx: Índices de prueba
    """
    # Cargar datos
    df = load_stock_data(file_path)
    
    # Calcular precio promedio
    df = calculate_average_price(df)
    
    # Obtener serie temporal
    average_prices = get_average_price_series(df)
    
    # Dividir datos
    train_data, val_data, test_data, indices = split_data(
        average_prices, train_ratio, val_ratio, test_ratio
    )
    
    if verbose:
        stock_name = os.path.basename(file_path)
        print_data_info(df, stock_name)
        print(f"División de datos:")
        print(f"  Entrenamiento: {len(train_data)} muestras ({train_ratio*100:.1f}%)")
        print(f"  Validación: {len(val_data)} muestras ({val_ratio*100:.1f}%)")
        print(f"  Prueba: {len(test_data)} muestras ({test_ratio*100:.1f}%)")
    
    # Retornar como diccionario
    return {
        'df': df,
        'average_prices': average_prices,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_idx': indices['train'],
        'val_idx': indices['val'],
        'test_idx': indices['test']
    }
