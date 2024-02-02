import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random

# Carregar os dados do arquivo CSV
data = pd.read_csv(
    'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/CSTR.csv')

# Separar os dados em texto e rótulos (classes)
text_data = data['Text']
labels = data['Class']

# Dividir os dados em treinamento e teste (80% para treinamento, 20% para teste)
text_train, text_test, labels_train, labels_test = train_test_split(
    text_data, labels, test_size=0.2, random_state=42)

# Criar uma representação vetorial dos dados de texto usando TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_train)

# Realizar a decomposição NMF
nmf = NMF(n_components=4, random_state=42)
nmf_features = nmf.fit_transform(tfidf_matrix)

# Identificar as classes únicas no conjunto de treinamento
unique_classes = labels_train.unique()

# Definir a classe positiva de forma aleatória
positive_class = random.choice(['class1', 'class2', 'class3', 'class4'])

# Atribuir rótulos às instâncias não rotuladas como positivas (1)
nmf_labels = labels_train.copy()
nmf_labels[nmf_labels != positive_class] = 0
nmf_labels[nmf_labels == positive_class] = 1

""" # Definir a classe positiva de forma aleatória
positive_class = random.choice(['class1', 'class2', 'class3', 'class4'])

# Atribuir rótulos às instâncias não rotuladas como positivas (1)
nmf_labels = labels_train.replace([positive_class], [1])
nmf_labels = nmf_labels.replace(
    ['class1', 'class2', 'class3', 'class4'], [0, 0, 0, 1])
 """
# Treinar um modelo de classificação usando regressão logística
classifier = LogisticRegression()
classifier.fit(nmf_features, nmf_labels)

# Transformar os dados de teste em sua representação vetorial e aplicar a decomposição NMF
tfidf_matrix_test = vectorizer.transform(text_test)
nmf_features_test = nmf.transform(tfidf_matrix_test)

# Prever as classes das instâncias de teste
predicted_labels = classifier.predict(nmf_features_test)

# Relatório de classificação
print(classification_report(labels_test, predicted_labels))
