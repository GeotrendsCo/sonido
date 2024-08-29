import pyodbc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import spacy

# Cargar el modelo de lenguaje español de spaCy
nlp = spacy.load("es_core_news_sm")

# Lista de palabras vacías en español e inglés
spanish_stop_words = [
    'un', 'una', 'unas', 'unos', 'uno', 'sobre', 'todo', 'también', 'tras', 'otro', 'algún', 'alguno', 'alguna',
    'algunos', 'algunas', 'ser', 'es', 'soy', 'eres', 'esos', 'esas', 'ese', 'aqui', 'estoy', 'estamos', 'esta', 'estais',
    'estan', 'como', 'en', 'para', 'detrás', 'ya', 'puede', 'puedo', 'por', 'qué', 'donde', 'quien', 'con', 'mi', 'mis',
    'tu', 'te', 'ti', 'nos', 'lo', 'los', 'las', 'el', 'la', 'si', 'no', 'siempre', 'siendo', 'fue', 'estaba', 'estaban',
    'estuve', 'estuvo', 'estado', 'he', 'has', 'ha', 'hemos', 'han', 'soy', 'es', 'son', 'eres', 'RAP'
]

english_stop_words = [
    'the', 'and', 'is', 'in', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'with', 'as', 'I', 'his', 'they', 
    'be', 'at', 'one', 'have', 'this', 'from', 'or', 'had', 'by', 'not', 'word', 'but', 'what', 'some', 'we', 'can', 
    'out', 'other', 'were', 'all', 'there', 'when', 'up', 'use', 'your', 'how', 'said', 'an', 'each', 'she', 'which', 
    'do', 'their', 'if', 'will', 'way', 'about', 'many', 'then', 'them', 'write', 'would', 'like', 'so', 'these', 
    'her', 'long', 'make', 'thing', 'see', 'him', 'two', 'has', 'look', 'more', 'day', 'could', 'go', 'come', 'did', 
    'number', 'sound', 'no', 'most', 'people', 'my', 'over', 'know', 'water', 'than', 'call', 'first', 'who', 'may', 
    'down', 'side', 'been', 'now', 'find'
]

# Combinar ambas listas
stop_words = spanish_stop_words + english_stop_words

# Función para lematizar el texto
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def exportar_datos_json(dfdatos, df_fut_prod_x_nodo, producto_df, file_path='datos_grafica.json'):
    # Combinar ambos dataframes
    combined_df = pd.concat([dfdatos, df_fut_prod_x_nodo])

    # Crear una lista de nodos y enlaces para D3.js
    nodes = []
    links = []

    # Crear un conjunto para evitar nodos duplicados
    unique_nodes = set()

    for i, row in combined_df.iterrows():
        # Determinar el tipo del nodo según la fuente de datos
        if row['nodo_id'] not in unique_nodes:
            if row['nodo_id'] in df_fut_prod_x_nodo['nodo_id'].values:
                node_type = 'futuro'  # Nodo que proviene de FUTXPRODXNOD
            else:
                node_type = 'nodo'  # Nodo que proviene de ProductoXNodo
            nodes.append({
                'id': row['nodo_id'],
                'name': row['nodo_nombre'],
                'type': node_type
            })
            unique_nodes.add(row['nodo_id'])

        if row['producto_id'] not in unique_nodes:
            nodes.append({
                'id': row['producto_id'],
                'name': producto_df.loc[producto_df['ID'] == row['producto_id'], 'NOMBRE'].values[0],
                'type': 'producto'  # Especificar que es un producto
            })
            unique_nodes.add(row['producto_id'])

        # Añadir enlaces
        links.append({
            'source': row['producto_id'],
            'target': row['nodo_id'],
            'value': row['similarity_score']
        })

    # Combinar nodos y enlaces en un diccionario
    data = {
        'nodes': nodes,
        'links': links
    }

    # Guardar los datos en un archivo JSON
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f'Datos exportados a {file_path}')

def procesar_productos_y_nodos(conn_str, similarity_threshold, fut_similarity_threshold, vectorizer):
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()

        # Borrar datos de las tablas intermedias
        cursor.execute("DELETE FROM ProductoXNodo")
        cursor.execute("DELETE FROM FUTXPRODXNOD")
        conn.commit()

        # Extraer datos de las tablas PRODUCTOFING y Nodo
        producto_query = "SELECT ID, NOMBRE FROM PRODUCTOFING"
        nodo_query = "SELECT ID, NOMBRE, DESCRIPCION, CAPACIDADES FROM Nodo"
        producto_x_nodo_query = "SELECT FKIDPRODUCTO, FKIDNODO FROM ProductoXNodo"
        fut_prod_x_nodo_query = "SELECT FKIDPRODUCTO, FKIDNODO FROM FUTXPRODXNOD"

        producto_df = pd.read_sql(producto_query, conn)
        nodo_df = pd.read_sql(nodo_query, conn)
        producto_x_nodo_df = pd.read_sql(producto_x_nodo_query, conn)
        fut_prod_x_nodo_df = pd.read_sql(fut_prod_x_nodo_query, conn)

        # Aplicar lematización
        producto_df['NOMBRE'] = producto_df['NOMBRE'].apply(lemmatize_text)
        nodo_df['DESCRIPCION'] = nodo_df['DESCRIPCION'].apply(lemmatize_text)
        nodo_df['CAPACIDADES'] = nodo_df['CAPACIDADES'].apply(lemmatize_text)

        # Crear un conjunto de pares existentes
        existing_pairs = set(tuple(x) for x in producto_x_nodo_df.values)
        existing_fut_pairs = set(tuple(x) for x in fut_prod_x_nodo_df.values)

        # Combinar los textos de productos y nodos para construir un vocabulario común
        combined_text = pd.concat([producto_df['NOMBRE'], nodo_df['DESCRIPCION'], nodo_df['CAPACIDADES']])

        # Crear la matriz TF-IDF combinada con los nuevos ajustes
        combined_tfidf = vectorizer.fit_transform(combined_text)

        # Separar las matrices TF-IDF para productos y nodos
        producto_tfidf = combined_tfidf[:len(producto_df)]
        nodo_tfidf = combined_tfidf[len(producto_df):len(producto_df)+len(nodo_df)]
        capacidades_tfidf = combined_tfidf[len(producto_df)+len(nodo_df):]

        # Calcular la similitud de coseno
        cosine_similarities_descripcion = cosine_similarity(producto_tfidf, nodo_tfidf)
        cosine_similarities_capacidades = cosine_similarity(producto_tfidf, capacidades_tfidf)

        # Crear la tabla intermedia ProductoXNodo
        producto_x_nodo = []

        # Llenar la tabla intermedia basándose en la similitud más alta
        for i, producto_id in enumerate(producto_df['ID']):
            for j, nodo_id in enumerate(nodo_df['ID']):
                similarity_score = max(cosine_similarities_descripcion[i, j], cosine_similarities_capacidades[i, j])
                if similarity_score >= similarity_threshold:
                    pair = (int(producto_id), int(nodo_id), similarity_score)
                    if pair[:2] not in existing_pairs:
                        producto_x_nodo.append(pair)

        # Crear el DataFrame con los datos de producto_x_nodo
        dfdatos = pd.DataFrame(producto_x_nodo, columns=['producto_id', 'nodo_id', 'similarity_score'])

        # Agregar los nombres de los nodos al DataFrame
        dfdatos = dfdatos.merge(nodo_df[['ID', 'NOMBRE']], left_on='nodo_id', right_on='ID').drop('ID', axis=1)
        dfdatos.rename(columns={'NOMBRE': 'nodo_nombre'}, inplace=True)

        # Insertar los datos en la tabla ProductoXNodo
        if producto_x_nodo:
            insert_query = "INSERT INTO ProductoXNodo (FKIDPRODUCTO, FKIDNODO, Similitud) VALUES (?, ?, ?)"
            cursor.executemany(insert_query, [(pair[0], pair[1], pair[2]) for pair in producto_x_nodo])

        # Productos no clasificados
        productos_no_clasificados = producto_df[~producto_df['ID'].isin(dfdatos['producto_id'])]

        # Crear la tabla intermedia FUTXPRODXNOD
        fut_prod_x_nodo = []

        # Llenar la tabla FUTXPRODXNOD basándose en la similitud de capacidades
        for i, producto_id in enumerate(productos_no_clasificados['ID']):
            for j, nodo_id in enumerate(nodo_df['ID']):
                similarity_score = cosine_similarities_capacidades[productos_no_clasificados.index[i], j]
                pair = (int(producto_id), int(nodo_id), similarity_score)
                if similarity_score >= fut_similarity_threshold and pair[:2] not in existing_fut_pairs:
                    fut_prod_x_nodo.append(pair)
                    existing_fut_pairs.add(pair[:2])  # Añadir par a los pares existentes para evitar duplicados en el bucle

        # Crear el DataFrame con los datos de fut_prod_x_nodo
        df_fut_prod_x_nodo = pd.DataFrame(fut_prod_x_nodo, columns=['producto_id', 'nodo_id', 'similarity_score'])

        # Agregar los nombres de los nodos al DataFrame
        df_fut_prod_x_nodo = df_fut_prod_x_nodo.merge(nodo_df[['ID', 'NOMBRE']], left_on='nodo_id', right_on='ID').drop('ID', axis=1)
        df_fut_prod_x_nodo.rename(columns={'NOMBRE': 'nodo_nombre'}, inplace=True)

        # Insertar los datos en la tabla FUTXPRODXNOD
        if fut_prod_x_nodo:
            insert_query = "INSERT INTO FUTXPRODXNOD (FKIDPRODUCTO, FKIDNODO, Similitud) VALUES (?, ?, ?)"
            cursor.executemany(insert_query, [(pair[0], pair[1], pair[2]) for pair in fut_prod_x_nodo])

        # Evaluar el resultado para determinar la mejor combinación de umbrales
        score = (len(dfdatos), len(df_fut_prod_x_nodo))
        return dfdatos, df_fut_prod_x_nodo, score, producto_df

# Parámetros de conexión
conn_str = (
    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
    r'DBQ=C:\Users\hernan.acosta\Desktop\SonidoSISPRO\ParaPython(2)1.accdb;'
)

best_combination = None
best_score = (0, 0)
best_dfdatos = None
best_df_fut_prod_x_nodo = None
best_producto_df = None

# Iterar sobre diferentes configuraciones de los parámetros de TfidfVectorizer y umbrales de similitud
for ngram_range in [(1, 1), (1, 2)]:
    for max_df in [0.85]:  # Solo probando con un valor representativo
        for min_df in [1, 5]:  # Reducción a solo dos valores
            for max_features in [None, 5000]:  # Reducción a solo dos valores
                for similarity_threshold in [0.10, 0.20, 0.30]:  # Manteniendo algunas opciones para similitud
                    for fut_similarity_threshold in [0.10, 0.20, 0.30]:  # Manteniendo algunas opciones para similitud
                        
                        # Crear el vectorizador TF-IDF con la configuración actual
                        vectorizer = TfidfVectorizer(
                            stop_words=stop_words,
                            ngram_range=ngram_range,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=max_features
                        )
                        
                        # Procesar productos y nodos con los umbrales actuales
                        dfdatos, df_fut_prod_x_nodo, score, producto_df = procesar_productos_y_nodos(
                            conn_str, similarity_threshold, fut_similarity_threshold, vectorizer)
                        
                        # Comparar para encontrar la mejor combinación
                        if score[0] > 0 and score[1] > 0:
                            if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] > best_score[1]):
                                best_score = score
                                best_combination = (ngram_range, max_df, min_df, max_features, similarity_threshold, fut_similarity_threshold)
                                best_dfdatos = dfdatos
                                best_df_fut_prod_x_nodo = df_fut_prod_x_nodo
                                best_producto_df = producto_df

print(f"La mejor combinación de parámetros es: {best_combination} con una puntuación de: {best_score}")

# Guardar la mejor combinación en un archivo Excel
if best_dfdatos is not None and best_df_fut_prod_x_nodo is not None:
    with pd.ExcelWriter('Mejor_Iteracion_version2.xlsx') as writer:
        best_dfdatos.to_excel(writer, sheet_name='ProductoXNodo', index=False)
        best_df_fut_prod_x_nodo.to_excel(writer, sheet_name='FUTXPRODXNOD', index=False)
        summary_df = pd.DataFrame({
            'N-gram Range': [best_combination[0]],
            'Max_df': [best_combination[1]],
            'Min_df': [best_combination[2]],
            'Max Features': [best_combination[3]],
            'Similarity Threshold': [best_combination[4]],
            'Future Similarity Threshold': [best_combination[5]]
        })
        summary_df.to_excel(writer, sheet_name='Parameters', index=False)

    # Exportar los datos a un archivo JSON para D3.js
    exportar_datos_json(best_dfdatos, best_df_fut_prod_x_nodo, best_producto_df)
