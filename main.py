import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import issparse
from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sys


## Generar la carga de todos los csv usados para las funciones!
# Funcion 1
df_funcion_PlaytimeGenre = pd.read_csv('df_funciones/df_funcion_PlaytimeGenre.csv')
# Funcion 2
df_funcion_UserForGenre = pd.read_csv('df_funciones/df_user_for_genre.csv')
# Funcion 3
users_reviews_fc3 = pd.read_csv('df_funciones/users_reviews_fc3.csv')
listado_juegos_sin_repetidos = pd.read_csv('df_funciones/listado_juegos_sin_repetidos.csv')
# Funcion 4
df_funcion_UsersWorstDeveloper = pd.read_csv('df_funciones/df_funcion_UsersWorstDeveloper.csv')
# Funcion 5
df_funcion_sentiment_analysis = pd.read_csv('df_funciones/df_funcion_sentiment_analysis.csv')


app = FastAPI()

@app.get("/")
async def root():
    return {"mensaje": "¡Bienvenido a mi API!"}

# Ejecutar la API con el comando: uvicorn main:app
# Controlar que este ubicado en la carpeta PI_MLOps_STEAM

## PRIMER FUNCIÓN: PLAY TIME GENRE

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str):

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

## Generar la carga de todos los csv usados para las funciones!

df_combinado2 = pd.read_csv("df_funciones/df_recomend_juego_fc6.csv")

@app.get('/RecomendacionJuego/{id_juego}')
def recomendacion_juego(id_juego: int):
    try:
        # Busca el juego en el DataFrame por ID
        juego_seleccionado = df_combinado2[df_combinado2['item_id'] == id_juego]

        # Verifica si el juego con el ID especificado existe
        if juego_seleccionado.empty:
            raise HTTPException(status_code=404, detail=f"No se encontró el juego con ID {id_juego}")

        # Obtener la primera fila de juego_seleccionado
        primera_fila = juego_seleccionado.iloc[[0]]

        title_game_and_genres = ' '.join(primera_fila['title'].fillna('').astype(str) + ' ' + primera_fila['genres'].fillna('').astype(str))

        tfidf_vectorizer = TfidfVectorizer()

        # Obtener las columnas 'title' y 'genres' antes de la concatenación
        chunk_tags_and_genres = df_combinado2[['title', 'genres']].fillna('').astype(str)

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











































# SEXTA FUNCION
# df_recomend_juego_fc6 = pd.read_csv("df_funciones/df_recomend_juego_fc6.csv")

## FUNCION 6 - VERSION 2
"""
@app.get('/RecomendacionJuego/{id_juego}')
def recomendacion_juego_conplaytime(id_juego: int, df_entrada = df_recomend_juego_fc6 ):
    '''
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
        {"title": "Juego A"},
        {"title": "Juego B"},
        {"title": "Juego C"},
        {"title": "Juego D"},
        {"title": "Juego E"}
    ]
    '''
    try:
        # Busca el juego en el DataFrame por ID
        juego_seleccionado = df_entrada[df_entrada['item_id'] == id_juego]

        # Verifica si el juego con el ID especificado existe
        if juego_seleccionado.empty:
            raise HTTPException(status_code=404, detail=f"No se encontró el juego con ID {id_juego}")

        # Genera una variable con el titulo y genero en tipo str
        # title_game_and_genres = ' '.join(juego_seleccionado['title'].fillna('').astype(str) + ' ' + juego_seleccionado['genres'].fillna('').astype(str))
        

        # Combina título, géneros y tiempo jugado del juego seleccionado en un solo texto
        title_game_genres_playtime = ' '.join(
            juego_seleccionado['title'].fillna('').astype(str) + ' ' +
            juego_seleccionado['genres'].fillna('').astype(str) + ' ' +
            juego_seleccionado['playtime_forever'].fillna('').astype(str)
        )

        # Información de títulos, géneros y tiempo jugado de todos los juegos
        chunk_tags_genres_playtime = (
            df_entrada['title'].fillna('').astype(str) + ' ' +
            df_entrada['genres'].fillna('').astype(str) + ' ' +
            df_entrada['playtime_forever'].fillna('').astype(str)
        )

        # Genera una cadena de texto con los titulos y los generos
        chunk_tags_genres_playtime = df_entrada['title'].fillna('').astype(str) + ' ' + df_entrada['genres'].fillna('').astype(str)
        
        # Genera una lista con dos elementos, el juego cargado y el resto de los juegos a comparar
        games_to_compare = [title_game_genres_playtime] + chunk_tags_genres_playtime.tolist()

        tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9)
        tfidf_matrix = tfidf_vectorizer.fit_transform(games_to_compare)

        # Generamos una limitacion truncando datos por un gran exceso de datos
        # Selecciona las primeras 15000 características
        tfidf_matrix_subset = tfidf_matrix[:15000, :]

        # Ahora a esos datos truncados toma los mejores 1000 campos y los compara con el dato de entrada
        svd = TruncatedSVD(n_components=1000)
        tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix_subset)

        # Calcula la similiridad de cosine
        # Calcula la matriz de similitud entre juego de entrada y resto de los juegos
        similarity_scores = cosine_similarity(tfidf_matrix_reduced)

        if similarity_scores is not None:
            # Ordena los indices similares de forma descendente y se toma desde el segundo para tomar los mas similares
            similar_games_indices = similarity_scores[0].argsort()[::-1]

            # Tomar las mejores 5 recomendaciones sin incluir el juego seleccionado
            num_recommendations = 5
            recommended_games = df_entrada.loc[similar_games_indices[1:num_recommendations + 1]]
            unique_recommendations = set()
            filtered_recommendations = []

            for index in similar_games_indices[1:]:
                game = df_entrada.loc[index, 'title']
                if game not in unique_recommendations:
                    unique_recommendations.add(game)
                    filtered_recommendations.append(game)

                if len(filtered_recommendations) >= num_recommendations:
                    break

            # Devolver la lista de juegos recomendados
            recommended_games = [{"title": game} for game in filtered_recommendations]

            return recommended_games

        return {"message": "No se encontraron juegos similares."}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

"""
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
