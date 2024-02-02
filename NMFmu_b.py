import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
df = pd.read_csv(
    'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/20ng.csv')

# Obter o número de tópicos/classes
num_topics = df['Class'].nunique()

# Pré-processar o texto
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Text'])

# Codificar as classes para facilitar o manuseio
le = LabelEncoder()
y = le.fit_transform(df['Class'])

# Configuração inicial
n_runs = 10
positive_class_ratio = 0.2

# Para armazenar as métricas
scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}


# Realize a execução do processo de treinamento e classificação dez vezes
for _ in range(n_runs):
    # Escolher a classe com mais exemplos para ser a positiva
    positive_class_label = df['Class'].value_counts().idxmax()
    positive_class_encoded = le.transform([positive_class_label])[0]

    # Criar novos rótulos onde a classe positiva é 1 e o resto é 0
    y_new = np.where(y == positive_class_encoded, 1, 0)

    # Crie um split estratificado para separar uma porcentagem dos exemplos positivos como rotulados
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=positive_class_ratio, random_state=42)
    for labeled_index, unlabeled_index in sss.split(X, y_new):
        X_labeled, X_unlabeled = X[labeled_index], X[unlabeled_index]
        y_labeled, y_unlabeled = y_new[labeled_index], y_new[unlabeled_index]

    # Aplicar NMF
    nmf = NMF(n_components=num_topics, solver='mu',
              beta_loss='kullback-leibler', l1_ratio=0.5, random_state=42)
    W = nmf.fit_transform(X_labeled)
    H = nmf.components_

    # Treinar o classificador
    clf = MultinomialNB()
    clf.fit(W, y_labeled)

    # Testar o classificador
    W_test = nmf.transform(X_unlabeled)
    y_pred = clf.predict(W_test)

    # Armazenar as métricas considerando todos os exemplos não rotulados
    scores['accuracy'].append(accuracy_score(y_unlabeled, y_pred))
    scores['precision'].append(precision_score(
        y_unlabeled, y_pred, average='micro', zero_division=0))
    scores['recall'].append(recall_score(
        y_unlabeled, y_pred, average='micro'))
    scores['f1'].append(f1_score(y_unlabeled, y_pred, average='micro'))

# Imprimir a média e o desvio-padrão das métricas
for metric in scores:
    print(
        f'{metric.capitalize()} - Média: {np.mean(scores[metric])}, Desvio Padrão: {np.std(scores[metric])}')
