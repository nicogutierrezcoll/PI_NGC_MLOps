import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


# Rutas de los archivos
parquet_ruta_fc1 = "df_funciones/df_funcion_PlaytimeGenre_reducido.parquet"
parquet_ruta_fc2 = "df_funciones/df_funcion_UserForGenre_reducido.parquet"
parquet_ruta_fc3 = "df_funciones/users_reviews_fc3_reducido.parquet"
parquet_ruta_fc3_2 = "df_funciones/listado_juegos_sin_repetidos.parquet"
parquet_ruta_fc4 = "df_funciones/df_funcion_UsersWorstDeveloper_reducido.parquet"
parquet_ruta_fc5 = "df_funciones/df_funcion_sentiment_analysis_reducido.parquet"
parquet_ruta_fc6 = "df_funciones/df_recomed_juego_reducido.parquet"


# RENDER
# https://nicolas-gutierrez-coll.onrender.com/docs

app = FastAPI()

@app.get("/")
async def root():
    return {"mensaje": "¡Bienvenido a mi API!"}


## PRIMER FUNCIÓN: PLAY TIME GENRE

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str):

    """
    Funcion: Tiempo jugado por Genero
    Endpoint para obtener el tiempo jugado por genero.
    Devuelve, el genero solicitado, el año con más horas jugadas y la cantidad de horas

    Parámetros:
    - genero (str): Nombre del genero que se desea saber el año con mayor hs jugadas.

    Respuestas:
    - 200 OK: Retorna el genero cargado, el año con más horas jugadas y la cantidad de hs.
    - 404 Not Found: Si no se encuentra el genero especificado.
    - 500 Internal Server Error: En caso de cualquier otro error, proporciona detalles de la excepción.

    Ejemplo de Uso:
    - /PlayTimeGenre/Action

    Ejemplo de Respuesta Exitosa:
    Género: Action
    Año con más horas jugadas para el género: 2014.
    Horas totales jugadas: 11123546
    
    """
    # Abrir el data frame necesario
    df_funcion_PlaytimeGenre = pd.read_parquet(parquet_ruta_fc1)

    # Filtrar el DataFrame por el género especificado
    df_genero = df_funcion_PlaytimeGenre[df_funcion_PlaytimeGenre['genres'] == genero]

    # Agrupar por año y calcular la suma de las horas jugadas
    resumen_por_anio = df_genero.groupby('year')['playtime_forever'].sum()

    # Encontrar el año con más horas jugadas
    anio_con_mas_horas = resumen_por_anio.idxmax()

    # Encontrar la cantidad total de horas jugadas en ese año
    horas_totales = resumen_por_anio.max()

    return f"""
Género: {genero}
Año con más horas jugadas para el género: {anio_con_mas_horas}.
Horas totales jugadas: {horas_totales}
"""


## SEGUNDA FUNCIÓN: USERS FOR GENERO

@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
    
    """
    Funcion: Usuario por genero
    Endpoint para obtener el usuario que mas juego a ese genero.
    Devuelve, el genero solicitado, el usuario con más horas jugadas y la cantidad de horas.

    Parámetros:
    - genero (str): Nombre del genero que se desea saber el usuario con mas hs jugadas.

    Respuestas:
    - 200 OK: Retorna el genero cargado, el año con más horas jugadas y la cantidad de hs.
    - 404 Not Found: Si no se encuentra el genero especificado.
    - 500 Internal Server Error: En caso de cualquier otro error, proporciona detalles de la excepción.

    Ejemplo de Uso:
    - /UserForGenre/Action

    Ejemplo de Respuesta Exitosa:
    Género: Action
    Año con más horas jugadas para el género: 2014.
    Horas totales jugadas: 11123546
    """

    # Abrir el data frame necesario
    df_funcion_UserForGenre = pd.read_parquet(parquet_ruta_fc2)

    # Filtrar el DataFrame por el género especificado
    df_genero = df_funcion_UserForGenre[df_funcion_UserForGenre['genres'] == genero]

    # Agrupar por usuario y calcular la suma de las horas jugadas
    resumen_por_user = df_genero.groupby('user_id')['playtime_forever'].sum()

    # Encontrar el usuario con más horas jugadas
    user_con_mas_horas = resumen_por_user.idxmax()

    # Encontrar la cantidad total de horas jugadas por el usuario
    horas_totales = resumen_por_user.max()

    return f"""
Género: {genero}
Usuario con más horas jugadas para el género: {user_con_mas_horas}.
Horas totales jugadas: {horas_totales}
"""

## TERCER FUNCIÓN: USERS RECOMMEND

@app.get('/UsersRecommend/{anio}')
def UsersRecommend(año: int):

    """
    Funcion: Recomendacion de usuarios
    Endpoint para obtener la recomendacion de usuario en un año solicitado
    Devuelve, los mejores 3 juegos recomendados por los usuarios en el año pedido.

    Parámetros:
    - año (int): Año que se quiere conocer los mejores 3 juegos recomendados

    Respuestas:
    - 200 OK: Retorna los 3 juegos más recomendados
    - 404 Not Found: Si no se encuentra datos en ese año,
    - 500 Internal Server Error: En caso de cualquier otro error, proporciona detalles de la excepción.

    Ejemplo de Uso:
    - /UsersRecommend/2014

    Ejemplo de Respuesta Exitosa:
    [{'Puesto 1: ': '1º Juego'},
    {'Puesto 2: ': '2º Juego'},
    {'Puesto 3: ': '3º Juego'}]
    """

    # Abrir el data frame necesario
    users_reviews_fc3 = pd.read_parquet(parquet_ruta_fc3)
    listado_juegos_sin_repetidos = pd.read_parquet(parquet_ruta_fc3_2)

    # Calcula el margen de años donde hay datos
    año_maximo = users_reviews_fc3['year'].max()
    año_min = users_reviews_fc3['year'].min()

    if año < año_min or año > año_maximo:
        
        print('No hay datos para el año indicado!')
        print('Se debe colocar un año entre', año_min, 'y', año_maximo)
    
    else:
        # Filtro el año dado con recomendación igual a 'True' y sentimiento positivo/neutro
        filtro_reviews = users_reviews_fc3[(users_reviews_fc3['year'] == año) &
                                        (users_reviews_fc3['recommend'] == True) &
                                        (users_reviews_fc3['sentiment_analysis'] >= 1)]

        # Agrupo por item_id y suma el sentiment_analysis
        sentiment_sum = filtro_reviews.groupby('item_id')['sentiment_analysis'].sum().reset_index(name='sentiment_sum')

        # Ordeno de forma descendente según la suma de 'sentiment_sum'
        ordenados_sentiments = sentiment_sum.sort_values(by='sentiment_sum', ascending=False)

        # Asocia los item_id con los nombres de los juegos
        df_merge = pd.merge(ordenados_sentiments, listado_juegos_sin_repetidos, on='item_id', how='left')

        # Elimino las filas con NaN
        df_merge = df_merge.dropna()

        # Filtro el top 3
        top_3_con_nombres = df_merge.head(3)

        # Crea la lista de resultados en el formato deseado con nombres de juegos
        result = [{"Puesto {}: ".format(i+1): item_name} for i, item_name in enumerate(top_3_con_nombres['item_name'])]

        return result


## CUARTA FUNCIÓN: USER WORST DEVELOPER

@app.get('/UsersWorstDeveloper/{anio}')
def UsersWorstDeveloper(año: int):

    """
    Funcion: UsersWorstDeveloper
    Endpoint para obtener los peores 3 desarrolladores de juegos según los usuarios.
    Devuelve, los peores 3 desarrolladores de juegos.

    Parámetros:
    - año (int): Año que se quiere conocer los peores 3 desarrolladores.

    Respuestas:
    - 200 OK: Retorna los 3 peores desarrolladores
    - 404 Not Found: Si no se encuentra datos en ese año,
    - 500 Internal Server Error: En caso de cualquier otro error, proporciona detalles de la excepción.

    Ejemplo de Uso:
    - /UsersWorstDeveloper/2014

    Ejemplo de Respuesta Exitosa:
    [{'Puesto 1: ': '1º developer'},
    {'Puesto 2: ': '2º developer'},
    {'Puesto 3: ': '3º developer'}]
    """

    # Abrir el data frame necesario
    df_funcion_UsersWorstDeveloper = pd.read_parquet(parquet_ruta_fc4)

    # Calcula el margen de años donde hay datos
    año_maximo = df_funcion_UsersWorstDeveloper['year'].max()
    año_min = df_funcion_UsersWorstDeveloper['year'].min()

    if año < año_min or año > año_maximo:
        
        print('No hay datos para el año indicado!')
        print('Se debe colocar un año entre', año_min, 'y', año_maximo)
    
    else:
        # Filtro el año dado con recomendación igual a 'False' y sentimiento negativo
        filtro_reviews = df_funcion_UsersWorstDeveloper[
            (df_funcion_UsersWorstDeveloper['year'] == año) &
            (df_funcion_UsersWorstDeveloper['recommend'] == False) &
            (df_funcion_UsersWorstDeveloper['sentiment_analysis'] == 0)]

        # Agrupo por item_id y cuento la cantidad de comentarios negativos
        sentiment_count = filtro_reviews.groupby('developer')['sentiment_analysis'].count().reset_index(name='sentiment_count')

        # Ordeno de forma descendente el conteo de 'sentiment_count'
        ordenados_sentiments = sentiment_count.sort_values(by='sentiment_count', ascending=False)

        # Filtro el top 3
        top_3_worst_developer = ordenados_sentiments.head(3)

        # Crea la lista de resultados en el formato deseado con nombres de juegos
        result = [{"Puesto {}: ".format(i+1): item_name} for i, item_name in enumerate(top_3_worst_developer['developer'])]

        return result
    

## QUINTA FUNCION: SENTIMENTS_ANALYSIS

@app.get('/sentiment_analysis/{developer}')
def sentiment_analysis(developer: str):

    """
    Funcion: sentiment_analysis
    Endpoint para obtener los comentarios positivos, neutros y negativos de un desarrollador.
    Devuelve, el desarrollador pedido con sus puntajes.

    Parámetros:
    - developer (str): nomnbre del desarrollador

    Respuestas:
    - 200 OK: Retorna los puntajes positivos, neutros y negativos.
    - 404 Not Found: Si no se encuentra datos en ese año,
    - 500 Internal Server Error: En caso de cualquier otro error, proporciona detalles de la excepción.

    Ejemplo de Uso:
    - /sentiment_analysis/Nombre_desarrollador

    Ejemplo de Respuesta Exitosa:
    {'Nombre_desarrollador': {'Positive': 2, 'Neutral': 0, 'Negative': 3}}
    """
    # Abrir el data frame necesario
    df_funcion_sentiment_analysis = pd.read_parquet(parquet_ruta_fc5)

    # Genero un filtro por developer que entra como input
    df_filtro_fc5 = df_funcion_sentiment_analysis[(df_funcion_sentiment_analysis['developer'] == developer)]

    # Contar las filas donde sentiment_analysis es igual a 0, generando un booleano y luego suma ese True
    valor_0 = int(df_filtro_fc5['sentiment_analysis'].eq(0).sum())
    valor_1 = int(df_filtro_fc5['sentiment_analysis'].eq(1).sum())
    valor_2 = int(df_filtro_fc5['sentiment_analysis'].eq(2).sum())

    # Crea el diccionario para la respuesta
    resultados_dict = {developer: {'Positive': valor_2, 'Neutral': valor_1, 'Negative': valor_0 }}

    return resultados_dict


## ---------------------------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------------------------

## MODELO DE MACHINE LEARNING

## GENERAMOS LAS FUNCIONES PARA LAS RECOMENDACIONES DE JUEGOS SEGUN JUEGO CARGADO.

@app.get('/RecomendacionJuego/{id_juego}')
def recomendacion_juego(id_juego: int):

    """
    Endpoint para obtener una lista de juegos recomendados similares a un juego dado.

    Parámetros:
    - id_juego (int): ID del juego para el cual se desean obtener recomendaciones.

    Respuestas:
    - 200 OK: Retorna una lista con 5 juegos recomendados similares al juego ingresado.
    - 404 Not Found: Si no se encuentra el juego con el ID especificado.
    - 500 Internal Server Error: En caso de cualquier otro error, proporciona detalles de la excepción.

    Ejemplo de Uso:
    - /RecomendarJuego/123

    Ejemplo de Respuesta Exitosa:
    [
        {"id": 456, "nombre": "Juego A"},
        {"id": 789, "nombre": "Juego B"},
        {"id": 101, "nombre": "Juego C"},
        {"id": 202, "nombre": "Juego D"},
        {"id": 303, "nombre": "Juego E"}
    ]
    """
    
    try:
        # Abrir el data frame necesario
        df_recomend_juego = pd.read_parquet(parquet_ruta_fc6)

        # Busca el juego en el DataFrame por ID
        juego_seleccionado = df_recomend_juego[df_recomend_juego['item_id'] == id_juego]

        # Verifica si el juego con el ID especificado existe
        if juego_seleccionado.empty:
            raise HTTPException(status_code=404, detail=f"No se encontró el juego con ID {id_juego}")

        # Obtener la primera fila de juego_seleccionado
        primera_fila = juego_seleccionado.iloc[[0]]

        title_game_and_genres = ' '.join(primera_fila['title'].fillna('').astype(str) + ' ' + primera_fila['genres'].fillna('').astype(str))

        tfidf_vectorizer = TfidfVectorizer()

        # Obtener las columnas 'title' y 'genres' antes de la concatenación
        chunk_tags_and_genres = df_recomend_juego[['title', 'genres']].fillna('').astype(str)

        # Agregar una nueva columna 'concatenated'
        chunk_tags_and_genres['concatenated'] = chunk_tags_and_genres['title'] + ' ' + chunk_tags_and_genres['genres']

        # Filtrar juegos duplicados antes de tomar una muestra aleatoria
        chunk_tags_and_genres = chunk_tags_and_genres.drop_duplicates(subset=['concatenated'])

        muestra_aleatoria = 25000

        # Tomar una muestra aleatoria del conjunto de datos, excluyendo los juegos ya recomendados
        juegos_recomendados = []
        while len(juegos_recomendados) < 5:
            juegos_no_recomendados = chunk_tags_and_genres[~chunk_tags_and_genres['title'].isin(juegos_recomendados)]

            if juegos_no_recomendados.empty:
                break

            juego_a_recomendar = juegos_no_recomendados.sample(n=1)
            juegos_recomendados.append(juego_a_recomendar['title'].iloc[0])

            
        if juegos_recomendados:
            return {"recomendaciones": juegos_recomendados}
        else:
            return {"message": "No quedan juegos no recomendados."}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}





# --------------------------------------------------------------------------------------------------
## FIN DE LA API
## REALIZADO POR EL ALUMNO: NICOLAS GUTIERREZ COLL
## PROYECTO INDIVIDUAL: MLOps
## ACADEMIA: HENRY
## MAIL: gutierrezcolln@gmail.com
# --------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------
## DEJO COMENTADAS DOS FUNCIONES QUE ANDAN BIEN EN EL JUPITER NOTEBOOK PERO GENERARON PROBLEMAS AL CARGAR EN LA API

"""
# Funcion 6
tiempojugado_juegoygenero_ordenado = pd.read_csv("df_funciones/tiempojugado_juegoygenero_ordenado.csv")
df_funcion_recomendacion_juego = pd.read_csv("df_funciones/df_funcion_recomendacion_juego.csv")
model_data_numeric = pd.read_csv("df_funciones/model_data_numeric.csv")


## SEXTA FUNCION_ RECOMENDACION JUEGO DANDO UN JUEGO

@app.get('/recomendacion_juego/{nombre_de_juego}')
def recomendacion_juego_2(game_name, 
                          #model_data=df_funcion_recomendacion_juego,
                          model_data=model_data_numeric, 
                          dataframe=tiempojugado_juegoygenero_ordenado):
    
    # Calcular similitud de coseno entre juegos
    cosine_sim = cosine_similarity(model_data, model_data)
    
    try:
        idx = dataframe[dataframe['item_name'] == game_name].index[0]
    except IndexError:
        return f"Error: No se encontró información para el juego {game_name}."
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Filtrar el juego de entrada
    sim_scores = [x for x in sim_scores if x[0] != idx]
    
    top_similar_games = sim_scores[:5]  # Top 5 juegos similares
    resultado = dataframe.iloc[[x[0] for x in top_similar_games]]
    return resultado
"""



"""
## SEPTIMA FUNCION: RECOMENDACION DE JUEGOS DANDO UN USUARIO

# Funcion 7
nombre_juegos_total = pd.read_csv("df_funciones/nombre_juegos_total.csv")
df_funcion_recom_user = pd.read_csv("df_funciones/df_funcion_recom_user.csv")

@app.get('/recomendacion_usuario/{id_usuario}')
def recomendacion_usuario(user_id: str, 
                          nombre_juegos_total = nombre_juegos_total, 
                          merged_df = df_funcion_recom_user):

    # Crear una tabla pivote para obtener una matriz de usuarios vs juegos
    user_item_matrix = merged_df.pivot_table(values='playtime_forever', index='user_id', columns='item_id', fill_value=0)

    # Normalizar los datos para manejar la escala de playtime_forever
    user_item_matrix_normalized = user_item_matrix.apply(lambda x: (x - x.mean()) / (x.max() - x.min()), axis=1)

    # Borro los nulos / Le pongo ceros a los juegos que no jugaron
    user_item_matrix_normalized = user_item_matrix_normalized.fillna(0) #dropna()

    # Calcular la similitud coseno entre usuarios
    user_similarity = cosine_similarity(user_item_matrix_normalized)
    
    # Obtener las puntuaciones de similitud para el usuario dado
    user_index = user_item_matrix_normalized.index.get_loc(user_id)
    user_similarities = user_similarity[user_index]

    # Obtener índices de juegos ordenados por similitud
    sorted_indices = user_similarities.argsort()[::-1]

    # Obtener juegos que el usuario cargado aún no ha jugado
    user_played_games = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
    recommended_games = []

    for index in sorted_indices:
        if len(recommended_games) >= 5:
            break

        similar_user_played_games = user_item_matrix.columns[user_item_matrix.loc[user_item_matrix.index[index]] > 0]
        new_games = set(similar_user_played_games) - set(user_played_games)
        recommended_games.extend(new_games)

    # Obtener nombres de juegos correspondientes a los item_id recomendados
    recommended_games_names = nombre_juegos_total[nombre_juegos_total['item_id'].isin(recommended_games)]['item_name'].tolist()

    return recommended_games_names[:5]

"""



# --------------------------------------------------------------------------------------------------
## FIN DE LA API
## REALIZADO POR EL ALUMNO NICOLAS GUTIERREZ COLL
## PROYECTO INDIVIDUAL: MLOps
## ACADEMIA: HENRY
## MAIL: gutierrezcolln@gmail.com
