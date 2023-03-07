# neural-padr-es-


import pandas as pd
from sklearn.svm import SVC
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# Carregar os dados de treinamento:
df = pd.read_csv('dados_treinamento.csv')
x_train = df['texto']
y_train = df['categoria']

# Criar o modelo SVM para classificação de texto:
text_classifier = SVC(kernel='linear')
text_classifier.fit(x_train, y_train)

# Criar o modelo LSTM para análise de sentimentos:
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=embedding_vector_length, input_length=50))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=64)

# Criar o modelo K-Means para agrupamento de informações:
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_train)

# Criar o modelo Random Forest para aprendizado supervisionado:
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
