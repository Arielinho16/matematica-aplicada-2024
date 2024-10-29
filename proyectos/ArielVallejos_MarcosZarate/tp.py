import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize, pos_tag
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Descargar las librerías necesarias de nltk si no las tienes
nltk.download("sentiwordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Cargar el dataset de tweets
tweets_df = pd.read_csv("test_data.csv")


# Función para limpiar el texto
def limpiar_texto(texto):
    texto = re.sub(r"http\S+", "", texto)  # Eliminar URLs
    texto = re.sub(r"@\w+", "", texto)  # Eliminar menciones
    texto = re.sub(r"#\w+", "", texto)  # Eliminar hashtags
    texto = re.sub(r"[^\w\s]", "", texto)  # Eliminar puntuación
    texto = texto.lower()  # Convertir a minúsculas
    return texto


# Función para obtener el puntaje de sentimiento de una palabra usando SentiWordNet
def obtener_sentimiento(palabra, tag):
    try:
        if tag.startswith("N"):  # Sustantivo
            synsets = list(swn.senti_synsets(palabra, "n"))
        elif tag.startswith("V"):  # Verbo
            synsets = list(swn.senti_synsets(palabra, "v"))
        elif tag.startswith("J"):  # Adjetivo
            synsets = list(swn.senti_synsets(palabra, "a"))
        elif tag.startswith("R"):  # Adverbio
            synsets = list(swn.senti_synsets(palabra, "r"))
        else:
            return 0, 0  # Si no es ninguna de las anteriores, no tiene puntaje

        if synsets:
            pos_score = synsets[0].pos_score()
            neg_score = synsets[0].neg_score()
            return pos_score, neg_score
        else:
            return 0, 0  # Si no hay synsets, devolver puntajes neutros
    except Exception as e:
        print(f"Error al obtener sentimiento para {palabra}: {e}")
        return 0, 0


# Función para analizar el sentimiento de un tweet completo
def analizar_sentimiento(texto):
    tokens = word_tokenize(texto)
    tagged_tokens = pos_tag(tokens)  # Etiquetado POS
    # Para almacenar los puntajes positivos y negativos acumulados de cada palabra del tweet.
    pos_total = 0
    neg_total = 0

    for token, tag in tagged_tokens:
        pos_score, neg_score = obtener_sentimiento(token, tag)
        pos_total += pos_score
        neg_total += neg_score

    return pos_total, neg_total


# Limpiar los tweets
tweets_df["Texto preprocesado"] = tweets_df["sentence"].apply(limpiar_texto)

# Aplicar el análisis de sentimiento a cada tweet
tweets_df["Sentimiento positivo"], tweets_df["Sentimiento negativo"] = zip(
    *tweets_df["Texto preprocesado"].apply(analizar_sentimiento)
)

# Revisar los tweets donde el análisis no ha salido como se esperaba
tweets_con_problemas = tweets_df[
    (tweets_df["Sentimiento positivo"] == 0) & (tweets_df["Sentimiento negativo"] == 0)
]

# Mostrar los resultados del Módulo 2
print("\nResultados del Módulo 2 (Puntajes positivos y negativos):")
print(tweets_df[["sentence", "Sentimiento positivo", "Sentimiento negativo"]].head(10))

# MODULO 3: Definir los conjuntos difusos usando skfuzzy

# Crear variables de entrada (puntajes positivos y negativos)
positivo = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "positivo")
negativo = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "negativo")

# Crear variable de salida (sentimiento)
sentimiento = ctrl.Consequent(np.arange(0, 10.1, 0.1), "sentimiento")

# Definir las funciones de membresía para positivo y negativo (bajo, medio, alto)
positivo["low"] = fuzz.trimf(positivo.universe, [0, 0, 0.5])
positivo["medium"] = fuzz.trimf(positivo.universe, [0, 0.5, 1])
positivo["high"] = fuzz.trimf(positivo.universe, [0.5, 1, 1])

negativo["low"] = fuzz.trimf(negativo.universe, [0, 0, 0.5])
negativo["medium"] = fuzz.trimf(negativo.universe, [0, 0.5, 1])
negativo["high"] = fuzz.trimf(negativo.universe, [0.5, 1, 1])

# Definir las funciones de membresía para la salida (neutral, positivo, negativo)
sentimiento["neutral"] = fuzz.trimf(sentimiento.universe, [3, 5, 7])
sentimiento["positive"] = fuzz.trimf(sentimiento.universe, [7, 8, 10])
sentimiento["negative"] = fuzz.trimf(sentimiento.universe, [0, 2, 3])

# MODULO 4: Reglas difusas

# Definir reglas difusas basadas en los conjuntos difusos
rule1 = ctrl.Rule(positivo["low"] & negativo["low"], sentimiento["neutral"])
rule2 = ctrl.Rule(positivo["medium"] & negativo["low"], sentimiento["positive"])
rule3 = ctrl.Rule(positivo["high"] & negativo["low"], sentimiento["positive"])
rule4 = ctrl.Rule(positivo["low"] & negativo["medium"], sentimiento["negative"])
rule5 = ctrl.Rule(positivo["medium"] & negativo["medium"], sentimiento["neutral"])
rule6 = ctrl.Rule(positivo["high"] & negativo["medium"], sentimiento["positive"])
rule7 = ctrl.Rule(positivo["low"] & negativo["high"], sentimiento["negative"])
rule8 = ctrl.Rule(positivo["medium"] & negativo["high"], sentimiento["negative"])
rule9 = ctrl.Rule(positivo["high"] & negativo["high"], sentimiento["neutral"])

# Crear el sistema de control difuso
sentiment_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
)
sentiment_simulation = ctrl.ControlSystemSimulation(sentiment_ctrl)

# MODULO 5: Aplicar las reglas y defuzzificación


# Función para evaluar el sentimiento difuso y defuzzificación
def evaluar_sentimiento_fuzzy(row):
    sentiment_simulation.input["positivo"] = row["Sentimiento positivo"]
    sentiment_simulation.input["negativo"] = abs(
        row["Sentimiento negativo"]
    )  # Usar el valor absoluto para el negativo
    sentiment_simulation.compute()  # Ejecutar la simulación
    return sentiment_simulation.output["sentimiento"]


# Aplicamos la evaluación de sentimiento difuso
tweets_df["Puntaje Defuzz"] = tweets_df.apply(evaluar_sentimiento_fuzzy, axis=1)


# Función para generar la descripción del sentimiento difuso
def obtener_sentimiento_difuso(row):
    if row["Puntaje Defuzz"] > 6.7:
        return "Positivo"
    elif row["Puntaje Defuzz"] < 3.3:
        return "Negativo"
    else:
        return "Neutral"


# Aplicamos la función para obtener la descripción textual del sentimiento difuso
tweets_df["Sentimiento difuso"] = tweets_df.apply(obtener_sentimiento_difuso, axis=1)


# **Clasificación final basada en el puntaje defuzzificado**
def clasificar_defuzz(row):
    if row["Puntaje Defuzz"] > 6.7:
        return 1  # Positivo
    elif row["Puntaje Defuzz"] < 3.3:
        return -1  # Negativo
    else:
        return 0  # Neutro


# Aplicamos la clasificación final
tweets_df["Clasificación final"] = tweets_df.apply(clasificar_defuzz, axis=1)


# Mostrar los resultados finales con el puntaje de defuzzificación y la clasificación
print("\nResultados del análisis de sentimiento difuso:")
print(
    tweets_df[
        [
            "sentence",
            "Sentimiento positivo",
            "Sentimiento negativo",
            "Sentimiento difuso",  # Columna de sentimiento difuso textual
            "Puntaje Defuzz",
            "Clasificación final",
        ]
    ].head(10)
)

# BENCHMARKS: Medición de tiempo y generación de CSV
resultados = []
tiempos_ejecucion = []

for index, row in tweets_df.iterrows():
    inicio_tiempo = time.time()  # Iniciar cronómetro

    # Proceso de análisis de sentimiento
    positivo = row["Sentimiento positivo"]
    negativo = row["Sentimiento negativo"]
    sentimiento_difuso = row["Sentimiento difuso"]
    clasificacion_final = row["Clasificación final"]

    # Detener cronómetro
    tiempo_total = time.time() - inicio_tiempo
    tiempos_ejecucion.append(tiempo_total)

    # Guardar los resultados
    resultados.append(
        [
            row["sentence"],
            row["sentiment"],
            positivo,
            negativo,
            sentimiento_difuso,
            tiempo_total,
        ]
    )

# Crear un DataFrame para exportar los resultados a CSV
resultados_df = pd.DataFrame(
    resultados,
    columns=[
        "Oración original",
        "Label original",
        "Puntaje Positivo",
        "Puntaje Negativo",
        "Resultado de inferencia",
        "Tiempo de ejecución",
    ],
)

# Guardar el archivo CSV
resultados_df.to_csv("resultados_benchmarks.csv", index=False)

# Calcular el tiempo promedio de ejecución total
tiempo_promedio_total = np.mean(tiempos_ejecucion)
print(f"\nTiempo promedio de ejecución total: {tiempo_promedio_total:.6f} segundos")

# Contar el total de tweets positivos, negativos y neutrales
total_positivos = len(tweets_df[tweets_df["Clasificación final"] == 1])
total_negativos = len(tweets_df[tweets_df["Clasificación final"] == -1])
total_neutrales = len(tweets_df[tweets_df["Clasificación final"] == 0])

# Mostrar los totales
print(f"\nTotal de tweets positivos: {total_positivos}")
print(f"Total de tweets negativos: {total_negativos}")
print(f"Total de tweets neutrales: {total_neutrales}")
