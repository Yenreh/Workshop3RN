# Informe Taller 3 - Redes Neuronales

## Descripción

Informe en formato LaTeX del Taller No. 3: Predicción de Series Temporales con Redes LSTM y GRU.

## Estructura del Informe

El informe contiene las siguientes secciones:

1. **Introducción**: Objetivos generales y específicos del taller
2. **Marco Teórico**: Fundamentos de LSTM, GRU y arquitecturas apiladas
3. **Metodología**: Descripción del dataset, preprocesamiento y configuraciones experimentales
4. **Resultados**: 
   - Experimentos con LSTM (6 configuraciones)
   - Experimentos con GRU (6 configuraciones)
   - Experimentos con modelos apilados (4 configuraciones)
   - Comparación LSTM vs GRU
   - Análisis de hiperparámetros
5. **Discusión**: Respuestas detalladas a las preguntas del taller
6. **Conclusiones**: Hallazgos principales, limitaciones y trabajos futuros
7. **Referencias**: Bibliografía citada

## Archivos

- `informe_taller3.tex` - Documento LaTeX principal
- `compile_report.sh` - Script para compilar el documento
- `README.md` - Este archivo

## Compilación

### Opción 1: Script automático (recomendado)

```bash
chmod +x compile_report.sh
./compile_report.sh
```

Este script:
1. Compila el documento LaTeX 2 veces (para referencias cruzadas)
2. Genera el PDF final
3. Limpia archivos auxiliares automáticamente

### Opción 2: Manual

```bash
pdflatex informe_taller3.tex
pdflatex informe_taller3.tex
```

## Requisitos

- LaTeX distribution (TeX Live o MiKTeX)
- Paquetes LaTeX requeridos (instalados automáticamente en la mayoría de distribuciones):
  - inputenc, babel
  - amsmath, amsfonts, amssymb
  - graphicx, geometry
  - fancyhdr, booktabs
  - listings, xcolor
  - hyperref, float

## Imágenes

El informe referencia gráficas ubicadas en:
- `../output/images/` (relativo al directorio del informe)

Gráficas incluidas:
- `lstm_comparison.png` - Comparación de configuraciones LSTM
- `gru_comparison.png` - Comparación de configuraciones GRU
- `global_comparison.png` - Comparación global de todos los modelos
- `seq_length_effect.png` - Efecto de longitud de secuencia
- `batch_size_effect.png` - Efecto del tamaño de batch
- `simple_vs_stacked.png` - Comparación modelos simples vs apilados

## Datos de Referencia

Los resultados cuantitativos se basan en el archivo:
- `../output/results.csv` - Métricas de todos los experimentos (16 modelos)

## Información del Estudiante

- **Nombre**: Herney Eduardo Quintero Trochez
- **Código**: 201528556
- **Universidad**: Universidad del Valle
- **Programa**: Ingeniería de Sistemas
- **Curso**: Redes Neuronales 2025-II

## Notas

- El informe está en español según requerimientos del curso
- Todas las cifras y métricas son exactas según `results.csv`
- Las gráficas se generaron automáticamente durante los experimentos
- Se recomienda revisar que todas las imágenes existan antes de compilar
