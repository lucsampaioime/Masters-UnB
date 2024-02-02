import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def nmf_positive_unlabeled_learning(csv_file, num_positive_examples):
    # Carregar o arquivo CSV
    df = pd.read_csv(csv_file)

    # Selecionar aleatoriamente a classe positiva e as classes não rotuladas
    classes = df['Class'].unique()
    positive_class = np.random.choice(classes)
    unlabeled_classes = classes[classes != positive_class]

    # Separar os exemplos positivos e não rotulados
    positive_examples = df[df['Class'] == positive_class].sample(
        n=num_positive_examples, random_state=42)
    unlabeled_examples = df[df['Class'].isin(unlabeled_classes)]

    # Converter o texto em recursos numéricos usando TF-IDF
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(df['Text'])
    positive_features = features[positive_examples.index]
    unlabeled_features = features[unlabeled_examples.index]

    # Aplicar NMF para aprendizado não supervisionado nos exemplos positivos
    nmf = NMF(n_components=1)  # Defina o número de componentes desejado
    positive_nmf_features = nmf.fit_transform(positive_features)

    # Calcular a similaridade entre exemplos positivos e não rotulados
    similarity_matrix = cosine_similarity(
        unlabeled_features, positive_nmf_features)

    # Identificar os exemplos não rotulados mais prováveis de serem negativos
    top_negative_indices = np.argsort(similarity_matrix[:, 0])[
        :num_positive_examples]
    top_negative_examples = unlabeled_examples.iloc[top_negative_indices]

    # Concatenar exemplos positivos e negativos mais prováveis
    labeled_data = pd.concat(
        [positive_examples, top_negative_examples], ignore_index=True)
    labels = np.array([1] * num_positive_examples +
                      [0] * num_positive_examples)

    # Converter o texto em recursos numéricos usando TF-IDF
    labeled_features = features[labeled_data.index]

    # Dividir os dados rotulados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_features, labels, test_size=0.2, random_state=42)

    # Treinar um classificador usando regressão logística
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Classificar os exemplos não rotulados usando o classificador treinado
    predicted_labels = classifier.predict(unlabeled_features)

    # Calcular as métricas de desempenho
    true_labels = np.zeros(len(unlabeled_examples))
    true_labels[:num_positive_examples] = 1
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Imprimir os resultados
    print("Classe positiva selecionada: ", positive_class)
    print("\nExemplos positivos rotulados:")
    print(positive_examples)
    print("\nExemplos não rotulados mais prováveis de serem negativos:")
    print(top_negative_examples)
    print("\nRótulos verdadeiros dos exemplos positivos:")
    print(true_labels[:num_positive_examples])
    print("\nRótulos previstos para os exemplos não rotulados:")
    print(predicted_labels)
    print("\nMétricas de desempenho:")
    print("Acurácia:", accuracy)
    print("Precisão:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


nmf_positive_unlabeled_learning(
    "C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/NMF/Datasets/CSTR.csv", 5)
