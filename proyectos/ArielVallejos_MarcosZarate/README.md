1- Instala las dependencias necesarias ejecutando el siguiente comando en el terminal desde el directorio del proyecto:

pip install -r requirements.txt

2- Descarga los datos necesarios de NLTK (solo la primera vez):

python -m nltk.downloader sentiwordnet punkt averaged_perceptron_tagger

3- Para ejecutar el análisis de sentimiento difuso, ejecuta el archivo principal tp.py:

python tp.py