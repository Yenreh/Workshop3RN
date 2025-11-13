"""
visualize.py
Módulo para visualización de datos y resultados de modelos.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_time_series(data, title="Serie Temporal", xlabel="Tiempo", 
                     ylabel="Valor", save_path=None):
    """
    Grafica una serie temporal.
    
    Args:
        data (np.ndarray or list): Datos a graficar
        title (str): Título del gráfico
        xlabel (str): Etiqueta del eje x
        ylabel (str): Etiqueta del eje y
        save_path (str): Ruta para guardar el gráfico (opcional)
    """
    plt.figure(figsize=(14, 6))
    plt.plot(data, linewidth=1.5, color='steelblue')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_average_price(df, stock_name="", save_path=None):
    """
    Grafica el precio promedio de una acción.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        stock_name (str): Nombre de la acción
        save_path (str): Ruta para guardar el gráfico (opcional)
    """
    if 'Average_Price' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Average_Price'")
    
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Average_Price'], linewidth=1.5, color='darkgreen')
    
    title = f"Precio Promedio - {stock_name}" if stock_name else "Precio Promedio"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio Promedio ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_data_splits(data, indices, labels=['Entrenamiento', 'Validación', 'Prueba'],
                    title="División de Datos", save_path=None):
    """
    Visualiza la división de datos en entrenamiento, validación y prueba.
    
    Args:
        data (np.ndarray): Datos completos
        indices (dict): Diccionario con índices de cada conjunto
        labels (list): Etiquetas para cada conjunto
        title (str): Título del gráfico
        save_path (str): Ruta para guardar el gráfico (opcional)
    """
    plt.figure(figsize=(14, 6))
    
    colors = ['steelblue', 'orange', 'green']
    
    for i, (key, (start, end)) in enumerate(indices.items()):
        plt.plot(range(start, end), data[start:end], 
                label=labels[i], linewidth=1.5, color=colors[i])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Índice', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_training_history(history, model_name="", save_path=None):
    """
    Grafica las curvas de pérdida durante el entrenamiento.
    
    Args:
        history: Objeto History de Keras o diccionario con historial
        model_name (str): Nombre del modelo
        save_path (str): Ruta para guardar el gráfico (opcional)
    
    Returns:
        fig: Objeto Figure de matplotlib
    """
    # Extraer datos del historial
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de pérdida
    axes[0].plot(history_dict['loss'], label='Entrenamiento', 
                linewidth=2, color='steelblue')
    axes[0].plot(history_dict['val_loss'], label='Validación', 
                linewidth=2, color='orange')
    axes[0].set_title('Pérdida (MSE)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Época', fontsize=11)
    axes[0].set_ylabel('Pérdida', fontsize=11)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de MAE
    axes[1].plot(history_dict['mae'], label='Entrenamiento', 
                linewidth=2, color='steelblue')
    axes[1].plot(history_dict['val_mae'], label='Validación', 
                linewidth=2, color='orange')
    axes[1].set_title('Error Absoluto Medio (MAE)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Época', fontsize=11)
    axes[1].set_ylabel('MAE', fontsize=11)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    if model_name:
        fig.suptitle(f'Historial de Entrenamiento - {model_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def plot_predictions(y_true, y_pred, model_name="", n_points=None, save_path=None):
    """
    Grafica las predicciones vs valores reales.
    
    Args:
        y_true (np.ndarray): Valores reales
        y_pred (np.ndarray): Predicciones
        model_name (str): Nombre del modelo
        n_points (int): Número de puntos a mostrar (opcional)
        save_path (str): Ruta para guardar el gráfico (opcional)
    
    Returns:
        fig: Objeto Figure de matplotlib
    """
    if n_points is not None:
        y_true = y_true[:n_points]
        y_pred = y_pred[:n_points]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico de series temporales
    axes[0].plot(y_true, label='Valor Real', linewidth=2, 
                color='darkgreen', alpha=0.7)
    axes[0].plot(y_pred, label='Predicción', linewidth=2, 
                color='red', alpha=0.7, linestyle='--')
    axes[0].set_title('Predicciones vs Valores Reales', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Muestra', fontsize=11)
    axes[0].set_ylabel('Precio ($)', fontsize=11)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de dispersión
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')
    
    # Línea de predicción perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Predicción perfecta')
    
    axes[1].set_title('Gráfico de Dispersión', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Valor Real ($)', fontsize=11)
    axes[1].set_ylabel('Predicción ($)', fontsize=11)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    if model_name:
        fig.suptitle(f'Resultados - {model_name}', 
                    fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def plot_residuals(y_true, y_pred, model_name="", save_path=None):
    """
    Grafica los residuos de las predicciones.
    
    Args:
        y_true (np.ndarray): Valores reales
        y_pred (np.ndarray): Predicciones
        model_name (str): Nombre del modelo
        save_path (str): Ruta para guardar el gráfico (opcional)
    
    Returns:
        fig: Objeto Figure de matplotlib
    """
    residuals = y_true - y_pred
    
    # Asegurar que residuals sea 1D para el histograma
    residuals_flat = residuals.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de residuos en el tiempo
    axes[0].plot(residuals_flat, linewidth=1, color='purple', alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Residuos en el Tiempo', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Muestra', fontsize=11)
    axes[0].set_ylabel('Residuo ($)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Histograma de residuos
    axes[1].hist(residuals_flat, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Distribución de Residuos', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Residuo ($)', fontsize=11)
    axes[1].set_ylabel('Frecuencia', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    if model_name:
        fig.suptitle(f'Análisis de Residuos - {model_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def plot_model_comparison(results_df, metric='mae', title=None, save_path=None):
    """
    Compara múltiples modelos visualmente.
    
    Args:
        results_df (pd.DataFrame): DataFrame con resultados de modelos
        metric (str): Métrica a comparar ('mae', 'rmse', 'r2', 'mape')
        title (str): Título personalizado del gráfico
        save_path (str): Ruta para guardar el gráfico (opcional)
    
    Returns:
        fig: Objeto Figure de matplotlib
    """
    fig = plt.figure(figsize=(14, 7))
    
    # Usar la columna 'Modelo' como etiquetas si existe
    if 'Modelo' in results_df.columns:
        models = results_df['Modelo'].tolist()
        # Buscar la métrica en las columnas
        if 'MAPE (%)' in results_df.columns and metric.lower() == 'mape':
            values = results_df['MAPE (%)'].values
            metric_label = 'MAPE (%)'
        elif 'R²' in results_df.columns and metric.lower() == 'r2':
            values = results_df['R²'].values
            metric_label = 'R²'
        elif metric.upper() in results_df.columns:
            values = results_df[metric.upper()].values
            metric_label = metric.upper()
        elif metric in results_df.columns:
            values = results_df[metric].values
            metric_label = metric
        else:
            raise ValueError(f"Métrica '{metric}' no encontrada en el DataFrame")
    else:
        models = results_df.index.tolist()
        values = results_df[metric].values
        metric_label = metric.upper()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
    
    # Añadir valores encima de las barras
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Modelo', fontsize=12, fontweight='bold')
    plt.ylabel(metric_label, fontsize=12, fontweight='bold')
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'Comparación de Modelos - {metric_label}', fontsize=14, fontweight='bold')
    
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def plot_all_metrics_comparison(results_df, save_path=None):
    """
    Compara todos los modelos en todas las métricas.
    
    Args:
        results_df (pd.DataFrame): DataFrame con resultados de modelos
        save_path (str): Ruta para guardar el gráfico (opcional)
    """
    metrics = ['mae', 'rmse', 'r2', 'mape']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in results_df.columns:
            models = results_df.index.tolist()
            values = results_df[metric].values
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
            bars = axes[idx].bar(range(len(models)), values, color=colors, 
                               alpha=0.8, edgecolor='black')
            
            # Añadir valores
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{value:.3f}', ha='center', va='bottom', 
                             fontsize=9, fontweight='bold')
            
            axes[idx].set_xlabel('Modelo', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(range(len(models)))
            axes[idx].set_xticklabels(models, rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Comparación Completa de Modelos', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()
