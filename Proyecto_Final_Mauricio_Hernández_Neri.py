import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# Cargar datos desde el archivo CSV
data = pd.read_csv('sent.csv', encoding='latin', header=None)

# Mostrar los primeros 10 registros del DataFrame
data.head(10)

# Asignar nombres a las columnas del DataFrame
data.columns = ['sentimiento', 'id', 'fecha', 'entrada', 'usuario', 'texto']

# Eliminar columnas innecesarias
data = data.drop(['id', 'fecha', 'entrada', 'usuario'], axis=1)

# Mostrar los primeros registros después de eliminar las columnas
data.head()

"""## Selección y visualización"""

# Mostrar los valores únicos de la columna 'sentimiento'
print(data['sentimiento'].unique())

# Crear una Serie con los valores de la columna 'sentimiento'
sentimiento_series = pd.Series(data['sentimiento'])

# Contar la frecuencia de cada valor en la Serie
print(sentimiento_series.value_counts())

# Seleccionar una muestra equilibrada de 600,000 registros por cada valor de 'sentimiento'
sdata = data.groupby('sentimiento').apply(lambda x: x.sample(600000)).reset_index(drop=True)

# Contar la frecuencia de cada valor en la columna 'sentimiento' después de seleccionar la muestra
print(sdata['sentimiento'].value_counts())

# Crear una Serie con los valores de 'sentimiento' en la muestra seleccionada
sent_series = pd.Series(sdata['sentimiento'])

# Contar la frecuencia de cada valor en la nueva Serie
value_counts = sent_series.value_counts()

# Colores para el gráfico de pastel
colors = ['#6E99A1', '#FFBC42']

# Crear un gráfico de pastel para visualizar la distribución de sentimientos
plt.figure(figsize=(8, 4))
value_counts.plot.pie(colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.ylabel('')
plt.title('Sentimientos positivos vs negativos')
plt.legend(['Negativo (0)', 'Positivo (4)'])

# Seleccionar 10 registros aleatorios de la muestra
random_rows = sdata.sample(n=10, random_state=50)

# Mostrar los sentimientos y los textos correspondientes de los registros seleccionados
for _, row in random_rows.iterrows():
    print(f"Sentimiento: {row['sentimiento']}\nTexto: {row['texto']}\n{'-' * 30}")

"""## Clasificación y limpieza general"""

# Mapeo de etiquetas de sentimiento
labels = {0: "Negativo", 4: "Positivo"}

# Función para decodificar las etiquetas de sentimiento
def label_decoder(label):
    return labels[label]

# Aplicar la función de decodificación a la columna 'sentimiento'
sdata.sentimiento = sdata.sentimiento.apply(lambda x: label_decoder(x))

# Palabras vacías (stop words) en inglés
stop_words = stopwords.words('english')

# Stemmer para reducir las palabras a su raíz
stemmer = SnowballStemmer('english')

# Expresión regular para limpiar el texto
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# Función para preprocesar el texto
def preprocess(text, stem=False):
    # Limpiar el texto utilizando la expresión regular
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        # Filtrar las palabras vacías (stop words)
        if token not in stop_words:
            if stem:
                # Aplicar el stemmer si se especifica
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    # Unir los tokens en un solo texto
    return " ".join(tokens)

# Aplicar la función de preprocesamiento a la columna 'texto'
sdata.texto = sdata.texto.apply(lambda x: preprocess(x))

"""## Visualización de diferencias en el texto"""

from wordcloud import WordCloud

# Generación de la nube de palabras para sentimientos negativos
plt.figure(figsize=(20, 20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800,colormap='cool').generate(" ".join(sdata[sdata.sentimiento == 'Negativo'].texto))
plt.imshow(wc , interpolation = 'bilinear')

# Generación de la nube de palabras para sentimientos positivos
plt.figure(figsize=(20, 20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800,colormap='cool').generate(" ".join(sdata[sdata.sentimiento == 'Positivo'].texto))
plt.imshow(wc , interpolation = 'bilinear')

# Generación de la nube de palabras para las palabras más frecuentes en sentimientos negativos
plt.figure(figsize=(10, 10))
negative_texts = sdata[sdata.sentimiento == 'Negativo'].texto
word_frequencies = negative_texts.str.split(expand=True).stack().value_counts()
top_15_words = word_frequencies.head(20)
wc = WordCloud(max_words=2000, width=800, height=400, colormap='cool').generate_from_frequencies(top_15_words)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Generación de la nube de palabras para las palabras más frecuentes en sentimientos positivos
plt.figure(figsize=(10, 10))
positive_texts = sdata[sdata.sentimiento == 'Positivo'].texto
word_frequencies = positive_texts.str.split(expand=True).stack().value_counts()
top_15_words = word_frequencies.head(20)
wc = WordCloud(max_words=2000, width=800, height=400, colormap='cool').generate_from_frequencies(top_15_words)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# Definición de tamaño de variables para la prueba
TRAIN_SIZE = 0.8
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 30

# División del conjunto de datos en entrenamiento y prueba
train_data, test_data = train_test_split(sdata, test_size=1-TRAIN_SIZE, random_state=42)
print("Tamaño de datos de entrenamiento:", len(train_data))
print("Tamaño de datos de prueba:", len(test_data))

from keras.preprocessing.text import Tokenizer

# Tokenización de los textos
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data.texto)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1

from keras.utils.data_utils import pad_sequences

# Padding de las secuencias de texto
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.texto), maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.texto), maxlen=MAX_SEQUENCE_LENGTH)

print("Forma de los datos de entrenamiento X:", x_train.shape)
print("Forma de los datos de prueba X:", x_test.shape)

# Codificación de las etiquetas
labels = train_data.sentimiento.unique().tolist()
encoder = LabelEncoder()
encoder.fit(train_data.sentimiento.to_list())

y_train = encoder.transform(train_data.sentimiento.to_list())
y_test = encoder.transform(test_data.sentimiento.to_list())

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("Forma de los datos de entrenamiento y:", y_train.shape)
print("Forma de los datos de prueba y:", y_test.shape)

#Descarga del modelo pre-entrenado
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

#Definiciones para incoporar el embedding al modelo usando el modelo de 300 dimensiones

GLOVE_EMB = 'glove.6B.300d.txt'
EMBEDDING_DIM = 300
LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 8
MODEL_PATH = 'best_model.hdf5'

# Carga de los vectores de palabras pre-entrenados GloVe
embeddings_index = {}
f = open(GLOVE_EMB)
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

print('Se encontraron %s vectores de palabras.' % len(embeddings_index))

#Definición de la matriz de embedding para encontrar los vectores asociados a las palabras presentes en el corpus.

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

# Capa de embeddings
embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                            EMBEDDING_DIM,
                                            weights=[embedding_matrix],
                                            input_length=MAX_SEQUENCE_LENGTH,
                                            trainable=False)

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

# Definición de la secuencia de entrada
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

# Capa de embeddings utilizando la matriz de embeddings pre-entrenados
embedding_sequences = embedding_layer(sequence_input)

# Capa de Dropout espacial
x = SpatialDropout1D(0.25)(embedding_sequences)

# Capa de convolución 1D
x = Conv1D(64, 5, activation='relu')(x)

# Capa Bidireccional LSTM
x = Bidirectional(LSTM(64, dropout=0.15, recurrent_dropout=0.3))(x)

# Capa completamente conectada con regularización L2
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

# Capa de Dropout
x = Dropout(0.35)(x)

# Capa completamente conectada con regularización L2
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

# Capa de salida
outputs = Dense(1, activation='sigmoid')(x)

# Definición del modelo
model = tf.keras.Model(sequence_input, outputs)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Compilación del modelo
model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',
              metrics=['accuracy'])

# Reducción de la tasa de aprendizaje durante el entrenamiento
reduce_lr = ReduceLROnPlateau(factor=0.2,
                              min_lr=0.01,
                              monitor='val_loss',
                              verbose=1)

# Entrenamiento del modelo
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(x_test, y_test), callbacks=[reduce_lr])

def decode_sentiment(score):
    return 'Positivo' if score > 0.5 else 'Negativo'

# Predicción de sentimientos en el conjunto de prueba
scores = model.predict(x_test, verbose=1, batch_size=10000)
y_pred_1d = [decode_sentiment(score) for score in scores]

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")

colors = ["#007BFF", "#FF8C00"]

# Gráfico de precisión del entrenamiento y validación
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento', color=colors[0])
plt.plot(history.history['val_accuracy'], label='Precisión de validación', color=colors[1])
plt.title('Precisión de entrenamiento y validación', fontsize=16)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()

# Gráfico de pérdida del entrenamiento y validación
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento', color=colors[0])
plt.plot(history.history['val_loss'], label='Pérdida de validación', color=colors[1])
plt.title('Pérdida de entrenamiento y validación', fontsize=16)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Pérdida', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Matriz de confusión
cm = confusion_matrix(test_data.sentimiento.to_list(), y_pred_1d)

# Matriz de confusión en porcentaje
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

class_labels = ['Negativo', 'Positivo']  # Reemplazar con las etiquetas de clase reales

# Gráfico de la matriz de confusión
plt.figure(figsize=(8, 6))
ax = plt.gca()

sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', cbar=False, ax=ax)

ax.set_xlabel('Predicho', fontsize=12)
ax.set_ylabel('Real', fontsize=12)

ax.set_xticks(np.arange(len(class_labels)) + 0.5)
ax.set_yticks(np.arange(len(class_labels)) + 0.5)
ax.set_xticklabels(class_labels, rotation=0, ha='center', fontsize=10)
ax.set_yticklabels(class_labels, rotation=0, ha='right', fontsize=10)

ax.set_title('Matriz de Confusión (%)', fontsize=14)

plt.tight_layout()
plt.show()

import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Reporte de clasificación
print(classification_report(list(test_data.sentimiento), y_pred_1d))

# Archivo original en: https://colab.research.google.com/drive/1IHhsiWzdj9WYrvxuCrRPkJyOUqqCe_TV?usp=sharing