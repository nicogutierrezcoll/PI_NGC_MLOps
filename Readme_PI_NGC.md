<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
# <h1 align=center> **REALIZADO POR NICOLAS GUTIERREZ COLL - DS-PT05** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

Datos del alumno: <br>
Nicolás Gutierrez Coll <br>
gutierrezcolln@gmail.com <br>
GitHub: nicogutierrezcoll <br>

Cohorte: Data Science - Part Time 05 

<hr>

REPOSITORIO DE GITHUB: https://github.com/nicogutierrezcoll/PI_NGC_MLOps.git

## **INFORMACIÓN A TENER EN CUENTA**
Drive con la carpeta de data.
Esta carpeta contiene los archivos pesados y completos de los datos de entrada, dataframe utilizados originalmete y los utilizados de forma intermedia en el proyecto.

Link: [DRIVE - Data] https://drive.google.com/drive/folders/1nesQQu6eog2E1QizHpzhs2pVosLs8Whc?usp=drive_link

## **Pasos del proyecto ejecutados**

## Contexto

El proyeto se fue generando desde el comienzo, partiendo desde la base de las 3 bases de datos obtenidas.
- australian_users_items
- output_steam_games
- australian_user_reviews

La solicitud principal fue lograr obtener un **MVP** (Minimum Viable Product) para el cierre del proyecto, haciendo un trabajo rapido de **Data Engineer**.

*Por lo tanto, se puso manos a la obra y se inicio el proyecto!!*


## EJECUCIÓN

## Primera Instancia: ETL

Se realizo la carga de los archivos a un DataFrame y se guardaron en formato .CSV para mayor comodidad en la lectura de los mismos cada vez que se trabajaba en el proyecto.

Una vez cargados, se empezo con el proceso de **`ETL`**.
Para cada archivo se genero una limpieza de datos, eliminación de campos nulos y duplicados. Se quitaron columnas innecesarias y se acomodaron los datos de las columnas que tenian listas o diccionarios de datos.

Al finalizar se generaron los 3 archivos base1 para arrancar como base para las siguientes necesidades.
- steam_games_base1.csv
- users_items_base1.csv
- users_reviews_base1.csv

Esto se puede ver en el archivo: `Proceso_ETL_archivos_Nico.ipynb`

## Segunda Instancia: Funciones

Partiendo de los archivos bases se siguio la limpieza de datos, el analizis de los mismos para generar los dataframe necesarios para poder construir las 5 funciones solicitadas.

Los DataFrame con las funciones son las siguientes:

1)  def PlayTimeGenre(genero: str)
    df: df_funcion_PlayTimeGenre.parquet

2)  def UserForGenre(genero: str)
    df: df_user_for_genre.parquet

3)  def UsersReviews(año: int)
    df: users_reviews_fc3.csv 
        listado_juegos_sin_repetir.csv

4)  def UsersWorstDeveloper.csv
    df: df_funcion_UsersWorstDeveloper.csv

5)  def SentimentAnalisis
    df: df_funcion_sentiment_analysis.csv

`Archivo: Funciones_Nico.ipynb`

*Aclaración: Los DataFrame indicados anteriormente, se refieren al proceso de trabajo. Posterior mente, por cuestiones de memoria se genero una muestra de los mismos y todos en formato parquet para reducir el peso de los mismos y que puedan ser ejecutados por Render.*


## Tercera Instancia: EDA y Modelo ML

En el EDA se empezo a analizar los datos y las relaciones de los mismos para buscar generar un modelo de recomendacion utilizando Machine Learning.

Analizando graficos se llego a la conclusion que se generara la relacion por genero ya que la misma es la que mayor insidencia tiene segun los graficos vistos.
Esto se puede ver en `Proceso_EDA_Nico.ipynb`
Mientras que los modelos de Machine Learning en `Modelo Machine Learning Nico.ipynb`.

Luego de varias pruebas y testeos se genero la funcion definitiva en el archivo `Funcion_6_ModeloML_Nico.ipynb`

Para esta instacia, se genero la siguiente funcion:

def Recomendacion_Juego(Id_juego: int)
    df: df_recomend_juego_fc6.parquet

*Aclaración: En un principio se genero las dos funciones solicitadas, tanto de recomendacion de juegos brindando juegos y la recomendacion de juegos brindando un usuario. Al pasarlos a la API, generaban una falla que no corrian, mientras que en el ipynb corrian bien y daban los resultados esperados. Por ese motivo se termino generando la función final.
Dichas funcines estan comentadas al final del "main.py".*

## Cuarta Instancia: API, Deployement y Render

En esta etapa, ya con las funciones listas pasamos a generar la API, para luego hacer el deployement con Render.

Para la API, se genero el archivo **`main.py`**, donde cargamos todas las funciones y los dataframe de cada funcion para que pueda correr correctamente.

Una vez corriendo la API, pasamos a hacer el deployement con Render.com

Para esta parte, por las limitaciones de espacio que ofrece render, genermos una muestras de todos los dataframe. Estos mismos son los que se encuentran en la carpeta "df_funciones".

Este paso, lo genermos con el archivo: `Reductor_tamaño_archivos_para_render.ipynb`

## Link de Render:
[Render]https://nicolas-gutierrez-coll.onrender.com


-----------------------------------

Alumno: Nicolás Gutierrez Coll <br>
gutierrezcolln@gmail.com <br>
Chorte: Data Science - PT05 <br>
Proyecto Individual: PI-MLOps <br>
Institución: Henry <br>
Diciembre 2023.<br>
