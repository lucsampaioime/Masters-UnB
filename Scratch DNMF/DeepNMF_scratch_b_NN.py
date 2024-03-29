from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class DeepNMF(nn.Module):
    def __init__(self, input_dim, topic_dim):
        super(DeepNMF, self).__init__()
        # Camadas que representam W e H
        self.document_topic = nn.Linear(input_dim, topic_dim, bias=False)
        self.topic_word = nn.Linear(topic_dim, input_dim, bias=False)

        # Inicialização não-negativa
        nn.init.xavier_uniform_(self.document_topic.weight)
        nn.init.xavier_uniform_(self.topic_word.weight)

    def forward(self, x):
        # A aplicação de ReLU garante a não-negatividade
        topic_representation = torch.relu(self.document_topic(x))
        word_representation = torch.relu(self.topic_word(topic_representation))
        return word_representation


def load_data_and_labels(filepath, n_labeled):
    # Carrega o dataset
    df = pd.read_csv(filepath)
    # Seleciona uma classe positiva aleatoriamente
    positive_class = np.random.choice(df['Class'].unique())
    df['Label'] = 0  # Inicializa todas as labels como negativas
    # Marca um número 'n_labeled' de documentos da classe positiva como positivos
    positive_indices = df[df['Class'] == positive_class].index
    labeled_indices = np.random.choice(
        positive_indices, size=n_labeled, replace=False)
    df.loc[labeled_indices, 'Label'] = 1

    vectorizer = TfidfVectorizer(max_features=1000)
    V = vectorizer.fit_transform(df['Text']).toarray()
    labels = df['Label'].values
    return torch.FloatTensor(V), torch.LongTensor(labels), positive_class


def train_model(V, labels, input_dim, topic_dim, epochs=100, lr=0.01):
    model = DeepNMF(input_dim, topic_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(V)
        loss = criterion(outputs, V)
        loss.backward()
        optimizer.step()

        # Calcula as métricas de desempenho
        if epoch % 10 == 0:
            with torch.no_grad():
                predicted_labels = outputs.argmax(dim=1)
                calculate_metrics(labels.numpy(), predicted_labels.numpy())
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


def calculate_metrics(true_labels, predicted_labels):
    # Ajuste para evitar o aviso de métricas indefinidas
    report = classification_report(
        true_labels, predicted_labels, output_dict=True, zero_division=0)
    print("Acurácia:", report['accuracy'])
    print("Precisão:", report['weighted avg']['precision'])
    print("Recall:", report['weighted avg']['recall'])
    print("F1 Score:", report['weighted avg']['f1-score'])


# Exemplo de execução
if __name__ == "__main__":
    V, labels, positive_class = load_data_and_labels(
        'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/Deep NMF/Datasets/CSTR.csv', n_labeled=10)
    input_dim = V.shape[1]  # Número de palavras (features)
    topic_dim = 10  # Número de tópicos desejado
    model = train_model(V, labels, input_dim, topic_dim, epochs=100, lr=0.01)
