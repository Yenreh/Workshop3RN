```
P á g i n a 1 | 2
```
# Facultad de Ingeniería

# Escuela de Ingeniería de Sistemas y Computación

```
Curso: Redes Neuronales
Taller No. 3 : Predicción promedio del valor de acciones del mercado con redes
LSTM y GRU
Semanas de realización: 5 noviembre – 15 noviembre de 2025
Fecha de entrega: 15 noviembre a través del Campus Virtual
RECUERDE QUE....
Como solución del taller se debe entregar un archivo comprimido nombrado:
Taller 3 _1reApellidoIntegrantesGrupo , que contiene un informe (.pdf) con una
explicación detallada del desarrollo del taller , los datos utilizados para realizar
las gráficas y los archivos fuente desarrollados (.ipynb or .py).
La práctica se puede realizar en grupos de máximo 3 personas.

disponible en el campus virtual
Correo electrónico: deisy.chaves@correounivalle.edu.co
```
# Predicción promedio del valor de acciones del mercado

# con redes LSTM y GRU

Las cotizaciones bursátiles en los mercados financieros son muy imprevisibles y volátiles.
Esto significa que no existen patrones coherentes en los datos que permitan modelizar los
precios de las acciones a lo largo del tiempo de forma casi perfecta. Por ello se hace
necesario el uso de arquitecturas de redes neuronales recurrentes con memoria (LSTM o
GRU) que permitan modelar la tendencia de los precios (si van a subir o bajar en un futuro
próximo), ver Figura 1. Dichos modelos ayudan a comparadores de acciones a decidir
cuándo comprarlas y/o vender sus acciones para obtener beneficios.

```
Figura 1. Ejemplo de predicción de precios de una acción en un mercado financiero.
Tomado de datacamp [https://www.datacamp.com/es/tutorial/lstm-python-stock-market]
```

```
P á g i n a 2 | 2
```
**1. (1.0 puntos)** Descargar datos correspondientes a los valores de una acción (por
    ejemplo, accer.u.txt) disponible en la carpeta Stocks del conjunto de datos “ **Huge**
    **Stock Market Dataset”** en **:** https://www.kaggle.com/datasets/borismarjanovic/price-
    volume-data-for-all-us-stocks-etfs. Cada archivo contiene para cada acción en una
    fecha (Date) dada:
       - Open: Precio de apertura del día
       - Close: Precio de cierre del día
       - High: Precio más alta de la acción en la fecha dada
       - Low: Precio más bajo de la acción en la fecha dada
       - Volume: número total de acciones negociadas de un valor/precio concreto
       - OpenInt: cantidad de acciones disponibles para la venta
    **a.** Grafique el valor promedio de la acción seleccionada ([High + Low] /2)
    **b.** Normalice los datos y cree ventanas de tiempo de **n** días correspondientes al valor
       promedio de la acción en el tiempo. **Indique el número n de días seleccionados**
    **c.** Divida los datos en tres conjuntos: entrenamiento, validación y prueba. **Indique los**
       **porcentajes seleccionados para la creación de los subconjuntos de datos
2. (1.5 puntos)** Entrene un modelo LSTM que permita la predicción del valor promedio de
    la acción seleccionada en el punto 1, 24 horas después del momento actual.
    **a.** Emplee diferentes parámetros (al menos 3 valores) de: número de neuronas
       recurrentes (seq_length), número de iteraciones y tamaño de batch. Así como
       habilitando/deshabilitando la opción de recurrent_dropout.
    **b.** Para cada caso gráfique la curva de perdida vs iteraciones, grafique las predicciones
       obtenidas (serie temporal verdadera vs predicciones con modelo LSTM) e indique
       el desempeño obtenido con la métrica deseada.
    **c.** Indique la configuración que genera mejores resultados, analice los resultados
       obtenidos y concluya
**3. (1.5 puntos)** Entrene un modelo GRU que permita la predicción del valor promedio de
    la acción seleccionada en el punto 1, 24 horas después del momento actual.
    **a.** Emplee diferentes parámetros (al menos 3 valores) de: número de neuronas
       recurrentes (seq_length), número de iteraciones y tamaño de batch. Así como
       habilitando/deshabilitando la opción de recurrent_dropout.
    **b.** Para cada caso gráfique la curva de perdida vs iteraciones, grafique las predicciones
       obtenidas (serie temporal verdadera vs predicciones con modelo GRU) e indique el
       desempeño obtenido con la métrica deseada.
    **c.** Indique la configuración que genera mejores resultados, analice los resultados
       obtenidos y concluya
**4.** ( **1 .0 puntos** ) Entrene una red de capas recurrentes apiladas (stacked recurrent layers
    network) que permita la predicción del valor promedio de la acción seleccionada en el
    punto 1, 24 horas después del momento actual.
       **a.** Indique el tipo de capa recurrente empleada (GRU/LSTM) y la configuración
          empleada: número de neuronas recurrentes (seq_length), número de iteraciones,
          número de capas apiladas.
       **b.** Gráfique la curva de perdida vs iteraciones, las predicciones obtenidas (serie
          temporal verdadera vs predicciones con modelo apilado, e indique el desempeño
          obtenido con la métrica deseada. Analice los resultados obtenidos y concluya.


