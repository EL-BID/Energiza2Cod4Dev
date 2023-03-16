<div id='Título-e-imagen-de-portada' />

# Energizados

***
<div id='insignias' />



<a target="_blank" href="https://colab.research.google.com/github/EL-BID/Energiza2Cod4Dev/blob/master/notebooks/colab_ejecucion_paso_paso.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Este proyecto presenta una guía para la detección de robo eléctrico basado en el aprendizaje automático supervisado utilizando una biblioteca de extracción de variables de series temporales junto con algoritmos de boosting y redes neuronales.

Ver [Notebook Paso a Paso](https://github.com/EL-BID/Energiza2Cod4Dev/blob/master/notebooks/ejecucion_paso_paso.ipynb)

***
<div id='índice' />

## Índice

* [Título](#Título-e-imagen-de-portada)

* [Insignias](#insignias)

* [Índice](#índice)

* [Descripción del proyecto](#descripción-del-proyecto)

* [Guía de instalación](#configuracion-ambiente)

* [Guía de usuario](#demostracion-del-proyecto)

* [Licencia](#licencia)

* [Limitación de responsabilidades](#limitación-de-responsabilidades)


***
<div id='descripción-del-proyecto' />

## Descripción del Proyecto
“Energizados” es un proyecto construido para mostrar cómo con el uso de aprendizaje automático se puede ayudar a detectar y disminuir las pérdidas no técnicas reduciendo tiempos de regularización e incrementando la precisión de identificación de fraudes.

El marco de detección de pérdidas no técnicas “Energizados” se puede dividir en 3 grandes etapas. La etapa de preprocesamiento de datos, la etapa de construcción de modelos simples basados en reglas o modelos baselines, luego el desarrollo de modelos más complejos como los supervisados y finalmente la etapa de evaluación de modelos.

<div style="width: 1000px; height: 600px;">
    <img src="img/Pryecto-Energiza2_V23.png" width="100%" height="100%">
</div>

### ***Etapa 1 : Preprocesamiento de datos / Exploración / Entendimiento***

Se requiere de esta etapa para tomar los datos en crudos y darles una estructura de datos significativa. Aquí es importante remarcar que un entendimiento y exploración de datos también es primordial ya que en este paso también se puede decidir que otras variables pueden usarse en el proceso de detección de pérdidas no técnicas. 

En nuestras pruebas, Energizados se evaluó en dos conjuntos de datos proveídos por dos empresas de distribución de energía. Donde se observó que además de las series de consumos mensuales, incorporar variables que describen a los usuarios también ayudan en la detección de usuarios fraudulentos. 

Entre otras cosas que se pueden observar en esta etapa es la proporción de usuario fraudulentos y no fraudulentos. En este tipo de problemas es común tener clases desbalanceadas, por lo general la proporción de usuarios fraudulentos no supera el 10%.

### ***Etapa 2 : Construcción de modelos***

En esta etapa primeramente se evaluaron modelos simples o modelos baselines para luego desarrollar modelos más complejos.

#### ***Modelos simples***

Modelos que a través de reglas analíticas pueden detectar un comportamiento anómalo en el consumo de energía de los usuarios. Estas reglas en general se derivan luego de hacer un análisis exploratorio de los datos y también de conocimientos de expertos.

- __Cambio o disminución en el consumo de energía:__  Regla que evalúa si un usuario disminuyó  dramáticamente su consumo actual con respecto a periodos anteriores.
- __Consumo constante:__ Regla que evalúa si el consumo fue constante por un periodo largo de tiempo.

#### ***Modelos supervisados***

En lo que respecta a la construcción de los modelos supervisados, se siguieron los siguientes pasos:

- Construcción de variables.
- Selección de variables.
- Manejo de datos desbalanceados.
- Optimización de hiperparámetros.
- Entrenamiento final, con los hiperparámetros encontrados en todo el conjunto de datos.


##### ***Ingeniería de variables***

La ingeniería de variables es el proceso de extracción y selección de las variables más importantes de los datos dados, normalmente se realiza para mejorar la capacidad de aprendizaje de los modelos en ML.

Esta etapa puede ser dividida en 2 subtareas, una tarea para la extracción o derivación de nuevas variables y otra tarea para la selección de las variables más importantes.

_Extracción de variables:_

En lo que respecta a los consumos mensuales de usuarios dados, estos datos por si solo carecen de características estadísticas que reflejen adecuadamente los patrones subyacentes en los datos de consumo de los usuarios, haciendo que los modelos de detección de robo de energía sean menos eficientes. 
Por lo tanto se crearon nuevas variables derivadas de los consumos mensuales de energía. Estas variables adicionales pueden ser divididas en 3 tipos: 

- __Estadísticas__: máximo, promedio, mínimo, mediana, son una muestra de los valores estadísticos calculados; 
- __Espectrales derivadas de la serie de consumo__: distancia de la señal, pendiente de la señal, varianza de la señal, etc.;
- __Temporales__: autocorrelación entre las variables, entropía, centroides, entre otros.

En lo que respecta a variables que caracterizan a los usuarios, por ejemplo, tipo de tarifa y actividad económica del usuario. De estas se derivaron variables a través de procesamiento clásicos de variables categóricas, entre los cuales se pueden destacar, creación de variables dummy, reducción de cardinalidad y encoding.

_Selección de variables:_

La selección de variables es el proceso de identificar un subconjunto representativo de variables de un grupo más grande.
En el desarrollo de los modelos, ejecutamos los siguientes pasos:

- Eliminación de variables constantes o con muy baja variabilidad.
- Eliminación de variables altamente correlacionadas entre sí.
- Selección de grupo de variables relevantes para la detección de robo.(Boruta)



##### ***Entrenamiento de modelos***

En cualquier técnica de aprendizaje automático supervisado, los datos etiquetados se proporcionan inicialmente al clasificador de aprendizaje para su propósito de entrenamiento. Luego, el modelo entrenado se evalúa por su capacidad para predecir y generalizar los datos no etiquetados de manera eficiente.

Como ocurre en la mayoría de los conjuntos de datos de detección de pérdidas no técnicas, estos están desbalanceados. Los datos desbalanceados generalmente se refieren a una situación que enfrentan los problemas de clasificación donde las clases no están representadas por igual.

Actualmente en la literatura se encuentran muchos métodos para poder abordar el problema de datos desbalanceados, además también existen paquetes software que automatizan el proceso y se pueden utilizar con python (Imbalearn).
En lo que respecta a lo que se usó en el siguiente marco de trabajo se utilizaron las  siguientes estrategias:

- Sobremuestreo (oversampling):  Consiste en generar nuevas muestras de la clase que está infrarrepresentadas.
- Sub–muestreo (under sampling): Consiste en eliminar ejemplos o filas de la clase mayoritaria. 

Finalmente, se utilizó la opción ‘under-sampling’, ya que se obtuvieron mejores resultados en la fase de desarrollo.

Para la optimización de hiperparámetros se siguió la estrategia de búsqueda aleatoria (Random Search), la cual consiste en muestrar valores posibles de los hiperparámetros y quedarse con aquellos que, al ser incluidos en el modelo, hayan generado mejores resultados en las métricas.

A continuación se da una breve descripción de los modelos usados.

***Light Gradient Boosting Machine (Light GBM)***

El modelo Light Gradient Boosting Machine (LGBM). LGBM, es un modelo dentro del “gradient boosting framework”, ya que usa algoritmos de aprendizaje basados en árboles de decisión.
En particular, LGBM es un modelo de ensamble (combinación de múltiples modelos) dado por el método de boosting. 

Este método comienza ajustando un modelo inicial en los datos y luego construye un segundo modelo que se enfoca en predecir con precisión los casos en los que el primer modelo tiene un desempeño deficiente. Se espera que la combinación de estos dos modelos sea mejor que cualquiera de los modelos por separado. Esta operación se puede hacer de manera sucesiva, construyendo varios modelos que reduzcan el error del anterior.

Algunas características que se destacan en LGBM son la velocidad de entrenamiento (light), la capacidad de manejar grandes volúmenes de datos utilizando poca memoria y el manejo de valores faltantes.

***Red Neuronal***

Las redes neuronales son sistemas de procesamiento de la información cuya estructura y funcionamiento están basados en las redes neuronales biológicas. Estos sistemas están compuestos por elementos simples denominados nodos o neuronas, los cuales se organizan por capas. 

Cada neurona está conectada con otra a partir de unos enlaces denominados conexiones, cada una de las cuales tiene un peso asociado. Estos pesos son valores numéricos modificables y representan la información que será usada por la red para resolver el problema que se presente.

Las conexiones entre las capas tienen un peso o valor asignado, el cual es importante para el aprendizaje de la red.

__Multicapa__

La red neuronal multicapa es una red donde todas las señales van en una misma dirección de neurona en neurona, esto se denomina feedforward.

En lo que respecta a la problemática de detección de fraude, las entradas son las variables con sus respectivos pre-procesamientos descritos en las secciones anteriores y la capa de salida nos da la probabilidad de que un usuario esté cometiendo fraude.

<div>
    <img src="img/multicapa.png" width="40%" height="40%">
</div>

__Concatenación LSTM - Multicapa__

Este es un tipo de red neuronal más compleja que la anterior, en éstas las neuronas presentan conexiones que pueden ir al siguiente nivel de neuronas y a su vez, conexiones que pueden ir al nivel anterior, se denominan conexiones feedforward y feedback.

Es una red neuronal que contiene ciclos internos que realimentan la red, generando así memoria. Dicha memoria le permite a la red aprender y generalizar a lo largo de secuencias de entradas en lugar de patrones individuales.

Se utilizan en problemas donde la secuencia de los datos sí importa, como por ejemplo en problemas de series de tiempo.
En lo que respecta al problema abordado, los consumos de energía mensuales de los usuarios se los puede tratar como una serie de tiempo.

En la siguiente figura se observa como se combinó una red lstm con una red multicapa para la detección de fraudes.

<div>
    <img src="img/LSTM.png" width="40%" height="40%">
</div>

### ***Etapa 3 : Evaluación de modelos***

En cualquier técnica de aprendizaje automático supervisado, los datos etiquetados se proporcionan inicialmente al clasificador de aprendizaje para su propósito de entrenamiento. Luego, el modelo entrenado se evalúa por su capacidad para predecir y generalizar los datos no etiquetados de manera eficiente.

El rendimiento de dichos modelos se evalúa en función de una serie de métricas de evaluación del rendimiento. En lo que respecta a este proyecto evaluamos en la siguiente métrica.

_Auc-roc_: La curva AUC - ROC es una medida de rendimiento para los problemas de clasificación que tiene en cuenta varios ajustes de umbral. La ROC es una curva de probabilidad y la AUC representa el grado o la medida de separabilidad. Indica en qué medida el modelo es capaz de distinguir entre clases. 

Cuanto más alto sea el AUC, mejor será el modelo para predecir 0s como 0s y 1s como 1s. Por analogía, cuanto mayor sea el AUC, mejor será el modelo para distinguir entre usuarios fraudulentos y aquellos que no lo son.

Esta curva tiene en el eje-y la medida TPR (True Positive Rate) y en el eje-x la medida FPR (False Positive Rate):
- TPR = Recall = TP / (TP+FN)
- FPR = FP / (FP+TN)

<div>
    <img src="img/roc_curve.png" width="20%" height="20%">
</div>

***

<div id='configuracion-ambiente' />

## Guía de instalación

El proyecto contiene la siguiente estructura de carpetas:
~~~
Energizado:
    |--- datos
    |--- notebooks
    |--- src :
    |        |--- modeling
    |        |--- preprocessing
    |        |--- helper
~~~

- datos :  contiene el conjunto de datos para poder ejecutar el código.
- notebooks : contiene las notebooks de ejecución. Existen dos notebooks una para ser ejecutada en Google-Colab y la otra para ejecutar en un entorno local.
- src : contiene los módulos python que dan soporte al proyecto.

Como se mencionó existen dos notebooks para poder ejecutar el marco de detección de pérdidas no técnicas en la distribución de energía. 

Si se quiere ejecutar el proyecto en Google-Colab seguir las instrucciones dentro de la notebook para colab.

Si se quiere ejecutar en un entorno local, seguir los siguientes pasos:

1. Tener instalada alguna distribución de Python 3.6 o superior.

2. Tener instalado jupyter lab.

3. Descarga/clonar el proyecto de github.

4. Lanzar jupyter lab ``` jupyter lab  ```

5. Posicionarse en la carpeta del proyecto.

6. Abrir un consola de comando en el entorno de jupyter lab.

7. Instalar los requerimientos utilizando ``` pip install -r requirements.txt  ```


***

<div id='demostracion-del-proyecto' />

## Guía de usuario 

Para hacer un demostración del uso de Energizados, compartimos un conjunto de datos anonimizado. El dataset está conformado de la siguiente manera.

- __Cantidad de registros__ : 42500
- __Cantidad de columnas__ : 19
- __% de fraudulentos__ : 5.8 %

Descripción de las columnas:

| Variable  | Descripción | Tipo de dato | Cardinalidad |
| :--- | :--- | :--- | :--- |
| Consumo de energía mensual | Indica el comportamiento de consumo a nivel mensual de los usuarios.  Se consideran los últimos 12 consumos.| Numérica | - |
| Actividad | Indica a qué actividad económica se dedica el usuario| Categoría | 284 |
| Tipo de Tarifa | Tarifa que tipo de tarifa se le cobra al usuario| Categoría | 47 |
| Tensión | Tensión instalada al usuario.| Categoría | 18 |
| Material instalacion | Indica tipo de material del medidor instalado| Categoría | 39 |
| Zona | Indica la ubicación geográfica a la que pertenece el usuario | Categoría | 38 |
| Target | Indica si el consumidor es fraudulento o no | Numérica | 0 - 1 |
| Fecha inspección | Indica la fecha en que se inspeccionó al usuario| Fecha | - |

Luego pare ver el código del marco de desarrollo funcionando compartimos una notebook donde se puede ir ejecutando paso a paso el proceso. 

Ver [Notebook Paso a Paso](https://github.com/EL-BID/Energiza2Cod4Dev/blob/master/notebooks/ejecucion_paso_paso.ipynb)


***

<div id='licencia' />

## Licencia 

El siguiente proyecto ha sido financiado por el BID. Ver la siguiente licencia [LICENCIA](https://github.com/EL-BID/Plantilla-de-repositorio/blob/master/LICENSE.md)


***

<div id='limitación-de-responsabilidades' />



## Limitación de responsabilidades

El BID no será responsable, bajo circunstancia alguna, de daño ni indemnización, moral o patrimonial; directo o indirecto; accesorio o especial; o por vía de consecuencia, previsto o imprevisto, que pudiese surgir:

i. Bajo cualquier teoría de responsabilidad, ya sea por contrato, infracción de derechos de propiedad intelectual, negligencia o bajo cualquier otra teoría; y/o

ii. A raíz del uso de la Herramienta Digital, incluyendo, pero sin limitación de potenciales defectos en la Herramienta Digital, o la pérdida o inexactitud de los datos de cualquier tipo. Lo anterior incluye los gastos o daños asociados a fallas de comunicación y/o fallas de funcionamiento de computadoras, vinculados con la utilización de la Herramienta Digital.

