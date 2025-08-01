# Clasificación Binaria con XGBoost en Apache Airflow

Este script permite implementar un flujo de trabajo escalable en Apache Airflow que realiza una clasificación binaria para predecir la pérdida (churn) de clientes usando XGBoost, procesando datos desde un archivo Parquet. El flujo incluye lógica condicional para manejar datasets pequeños y grandes, aprovechando las capacidades de procesamiento distribuido. Cada sección del script incluye comentarios extensos en español y explicaciones breves a continuación de las líneas de código.

## Requisitos Previos

Se debe tener instalado:

- [Apache Airflow](https://airflow.apache.org/docs/apache-airflow/stable/start.html) (`pip install apache-airflow`)
 
- [XGBoost](https://xgboost.readthedocs.io/en/stable/) (`pip install xgboost`)

- [Pandas](https://pandas.pydata.org/) (`pip install pandas`)

- [PyArrow](https://arrow.apache.org/docs/python/) (`pip install pyarrow`)

- [Dask](https://docs.dask.org/en/stable/) (`pip install dask[dataframe]`), para procesamiento distribuido de grandes volúmenes de datos

También se debe tener un archivo Parquet con los datos de churn, que incluya una columna objetivo binaria ("churn": 0 = no churn, 1 = churn).

## Estructura del flujo DAG de Airflow

1. Carga de datos: Leer el archivo Parquet, eligiendo entre Pandas o Dask según el tamaño del archivo.

2. Preprocesamiento: Limpieza y preparación de datos.

3. Entrenamiento: Ajuste del modelo XGBoost.

4. Evaluación: Métricas de desempeño.
\5. Almacenamiento de resultados: Guarda el modelo entrenado y el reporte de métricas.

## Script Python del DAG de Airflow

```
python name=dags/xgboost_churn_airflow.py
```
## Sección 1: Importación de librerías y configuración

En esta sección se importan todas las librerías necesarias para ejecutar el flujo de trabajo en Airflow, así como las herramientas para procesar datos, entrenar modelos y distribuir la carga de trabajo. Esta estructura modular permite escalar el flujo según las necesidades del dataset y facilita el mantenimiento y la extensibilidad.
```
import os  	# Para operaciones con archivos y rutas
import pandas as pd  	# Para manipulación de datos en memoria
import dask.dataframe as dd  	# Procesamiento distribuido de datos grandes
import xgboost as xgb  	# Para el modelo de clasificación XGBoost
from sklearn.model_selection import train_test_split  	# Para dividir el dataset
from sklearn.metrics import classification_report  	# Para evaluar el modelo
from airflow import DAG  	# Para definir y gestionar el DAG en Airflow
from airflow.operators.python import PythonOperator  	# Para crear tareas Python
from datetime import datetime  	# Para definir fechas de inicio del DAG
import joblib  	# Para guardar el modelo entrenado
```
### Variables globales
```
PARQUET_PATH = '/path/to/churn_data.parquet'  	# Ruta al archivo Parquet
TARGET_COLUMN = 'churn'  	# Nombre de la columna objetivo
MODEL_PATH = '/tmp/churn_xgb_model.pkl'  	# Ruta para guardar el modelo
REPORT_PATH = '/tmp/churn_classification_report.txt'  	# Ruta para el reporte
SPLIT_SIZE_MB = 100                                                    # Umbral para decidir si usar Pandas o Dask (en MB)
```
## Sección 2: Definición del DAG de airflow

Aquí se define el DAG principal, asignándole un identificador único, la política de ejecución y otros parámetros. Esto permite a Airflow orquestar y monitorizar el flujo de trabajo de manera automática y repetible.

```
default_args = {
    'owner': 'airflow',  	# Dueño del DAG
    'retries': 1,  	# Número de reintentos en caso de fallo
}

with DAG(
    dag_id='xgboost_churn_classification',  	# Identificador del DAG
    default_args=default_args,  	# Argumentos por defecto
    start_date=datetime(2023, 1, 1),  	# Fecha de inicio
    schedule_interval=None,  	# Ejecución manual
    catchup=False  	# No ejecutar tareas pasadas automáticamente
) as dag:
```
    
## Sección 3: Función de carga de datos

Esta función decide automáticamente si cargar los datos en memoria usando Pandas para archivos pequeños, o bien utilizar Dask para procesamiento distribuido en archivos grandes. Esta lógica condicional permite escalar el flujo sin modificar el código.
```
    def load_data(kwargs):
        ```
        Carga el archivo Parquet usando Pandas o Dask según su tamaño.
        Devuelve un diccionario serializado para ser pasado entre tareas.
        ```
        file_size = os.path.getsize(PARQUET_PATH) / (1024 * 1024)  	# Tamaño en MB
        if file_size < SPLIT_SIZE_MB:  	# Si el archivo es pequeño
            df = pd.read_parquet(PARQUET_PATH)  	# Carga con Pandas
            backend = 'pandas'  	# Marca el backend utilizado
        else:  	# Si el archivo es grande
            df = dd.read_parquet(PARQUET_PATH)  	# Carga con Dask
            backend = 'dask'  	# Marca el backend utilizado
        kwargs['ti'].xcom_push(key='backend', value=backend)  	# Guarda el backend en XCom
        kwargs['ti'].xcom_push(key='data_shape', value=str(df.shape))  # Guarda la forma de los datos
        df_head = df.head(100) if backend == 'dask'
                                                  else df.head()                        # Muestra las primeras filas para diagnóstico
        kwargs['ti'].xcom_push(
                                    key='data_head', value=df_head.to_json())  	# Guarda una muestra en XCom
        df.to_parquet('/tmp/churn_tmp.parquet')  # Guarda una copia temporal para las siguientes tareas

    load_data_task = PythonOperator(
        task_id='load_data',  	# Nombre de la tarea en Airflow
        python_callable=load_data,  	# Función a ejecutar
        provide_context=True,  	# Permite acceso a contexto de Airflow
    )
```    
## Sección 4: Función de preprocesamiento

En esta sección se realiza la limpieza básica y la preparación de los datos. Se eliminan valores nulos y se separan las variables predictoras de la variable objetivo. Además, se utiliza lógica condicional para tratar tanto datos en memoria como distribuidos.
```    
    def preprocess_data(kwargs):
        backend = kwargs['ti'].xcom_pull(key='backend')  	# Recupera el backend usado
        if backend == 'pandas':  	# Si se usó Pandas
            df = pd.read_parquet('/tmp/churn_tmp.parquet')  	# Carga en Pandas
            df = df.dropna()  	# Elimina filas con valores nulos
        else:  	# Si se usó Dask
            df = dd.read_parquet('/tmp/churn_tmp.parquet')  	# Carga en Dask
            df = df.dropna()  	# Elimina valores nulos (distribuido)
            df = df.compute()                                                              # Convierte a Pandas para entrenamiento
        X = df.drop(TARGET_COLUMN, axis=1)  	# Variables predictoras
        y = df[TARGET_COLUMN]  	# Variable objetivo
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )  # Divide en entrenamiento y prueba
        X_train.to_parquet('/tmp/churn_X_train.parquet')  	# Guarda conjunto de entrenamiento
        X_test.to_parquet('/tmp/churn_X_test.parquet')  	# Guarda conjunto de prueba
        y_train.to_frame().
             to_parquet('/tmp/churn_y_train.parquet')  	# Guarda etiquetas de entrenamiento
        y_test.to_frame().to_parquet('/tmp/churn_y_test.parquet')  	# Guarda etiquetas de prueba

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        provide_context=True,
    )
```

## Sección 5: Función de entrenamiento

Se ajusta el modelo XGBoost para clasificación binaria utilizando los datos procesados. Se configuran parámetros estándar y se entrena el modelo, guardándose para uso futuro o despliegue.
```    
    def train_model(kwargs):
        X_train = pd.read_parquet('/tmp/churn_X_train.parquet')  	# Carga X de entrenamiento
        y_train = pd.read_parquet(
                '/tmp/churn_y_train.parquet')[TARGET_COLUMN]  		# Carga y de entrenamiento
        model = xgb.XGBClassifier(
            objective='binary:logistic',  		# Clasificación binaria
            eval_metric='logloss',  		# Métrica de evaluación
            use_label_encoder=False,  		# Evita advertencia
            n_jobs=-1,  	# Uso de todos los núcleos disponibles
            random_state=42  		# Reproducibilidad
        )  # Instancia del modelo
        model.fit(X_train, y_train)  		# Entrenamiento del modelo
        joblib.dump(model, MODEL_PATH)  		# Guarda el modelo entrenado

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True,
    )
```

## Sección 6: Función de evaluación y reporte

Se evalúa el desempeño del modelo usando el conjunto de prueba, generando un reporte de métricas estándar (precisión, recall, f1-score). El reporte se almacena para revisión posterior.
```    
    def evaluate_model(kwargs):
        X_test = pd.read_parquet('/tmp/churn_X_test.parquet')  			# Carga X de prueba
        y_test = pd.read_parquet(
                         '/tmp/churn_y_test.parquet')[TARGET_COLUMN]  		# Carga y de prueba
        model = joblib.load(MODEL_PATH)  			# Carga el modelo entrenado
        y_pred = model.predict(X_test)  			# Predicciones
        report = classification_report(y_test, y_pred)  			# Genera reporte
        with open(REPORT_PATH, 'w') as f:
            f.write(report)  		# Guarda el reporte en un archivo

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
    )
```
## Definición del orden de las tareas en el DAG
```
load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task    # Encadenamiento
```

# Resumen y Recomendaciones de Uso

Este flujo de trabajo en Airflow permite procesar y modelar datos de churn de clientes usando XGBoost de forma escalable y automatizada, adaptándose a distintos tamaños de archivo. Aprovecha la integración de Dask para grandes volúmenes y Pandas para datasets manejables en memoria. El modelo y las métricas quedan almacenados para su revisión y despliegue.

## Recomendaciones:

•  Personalizar el preprocesamiento según las características de los datos reales.

• Ajustar los hiperparámetros de XGBoost para obtener mejores resultados en el problema específico.

• Integrar notificaciones o almacenamiento en la nube como pasos adicionales en el DAG.

• Usar [Airflow Variables y Connections](https://airflow.apache.org/docs/apache-airflow/stable/howto/variable.html) para mayor flexibilidad y seguridad.

• Consultar la [documentación oficial de Airflow](https://airflow.apache.org/docs/apache-airflow/stable/index.html), y [XGBoost](https://xgboost.readthedocs.io/en/stable/) y [Dask](https://docs.dask.org/en/stable/) para profundizar en ajustes avanzados y mejores prácticas.


## Fuentes

•   [Apache Airflow Docs](https://airflow.apache.org/docs/apache-airflow/stable/)
•   [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
•   [Dask for Big Data](https://docs.dask.org/en/stable/)
•   [Pandas Parquet](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html)
•   [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

