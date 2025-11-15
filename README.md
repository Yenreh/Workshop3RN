# Workshop 3 - Predicción de Series Temporales con LSTM y GRU

Predicción de precios de acciones utilizando redes neuronales recurrentes (LSTM y GRU).

## Instalación

### 1. Crear entorno conda con Python 3.10

```bash
conda create -n workshop3 python=3.10
conda activate workshop3
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
Workshop3/
├── src/
│   ├── Taller3.ipynb          # Notebook principal con todos los experimentos
│   ├── data_loader.py         # Carga y preparación de datos
│   ├── preprocessing.py       # Normalización y creación de secuencias
│   ├── models.py              # Arquitecturas LSTM/GRU (simple y apiladas)
│   ├── utils.py               # Callbacks, evaluación y guardado
│   ├── visualize.py           # Gráficas y visualizaciones
│   ├── train_lstm.py          # Script independiente para LSTM
│   ├── train_gru.py           # Script independiente para GRU
│   ├── train_stacked.py       # Script para modelos apilados
│   ├── dataset/
│   │   └── Stocks/            # Archivos .txt con datos de acciones
│   │       ├── amd.us.txt
│   │       ├── aapl.us.txt
│   │       └── ...
│   └── output/                # Generado al ejecutar
│       ├── models/            # Modelos entrenados (.keras)
│       ├── images/            # Gráficas generadas (.png)
│       └── results.csv        # Tabla con todas las métricas
├── requirements.txt           # Dependencias del proyecto
└── README.md
```

## Uso

### Ejecutar Notebook

```bash
cd src
jupyter notebook Taller3.ipynb
```

O con JupyterLab:

```bash
cd src
jupyter lab Taller3.ipynb
```

### Configuración GPU/CPU

En la primera celda del notebook:

```python
USE_CPU = False  # True para forzar CPU, False para intentar GPU
```

## Estructura del Notebook

El notebook `Taller3.ipynb` contiene:

1. **Configuración e importaciones** - Setup inicial
2. **Carga y exploración de datos** - Dataset AMD
3. **Experimentos LSTM** - 6 configuraciones diferentes
4. **Experimentos GRU** - 6 configuraciones diferentes
5. **Modelos apilados** - 4 configuraciones (2 y 3 capas)
6. **Análisis comparativo** - Resultados y gráficas

**Total**: 16 experimentos con análisis completo.

## Resultados

Al ejecutar el notebook se generan:

- **Modelos entrenados**: `src/output/models/*.keras`
- **Gráficas**: `src/output/images/*.png`
- **Métricas CSV**: `src/output/results.csv`

## Dataset

**Acción**: AMD (Advanced Micro Devices)  
**Archivo**: `src/dataset/Stocks/amd.us.txt`

Para cambiar la acción, editar en el notebook (celda 6):

```python
STOCK_FILE = 'dataset/Stocks/otra_accion.us.txt'
```

## Autor

Herney Eduardo Quintero Trochez  
Universidad del Valle - Redes Neuronales 2025-II
