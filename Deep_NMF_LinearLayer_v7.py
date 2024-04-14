from matplotlib import pyplot as plt
from matplotlib.pyplot import text
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import sklearn.decomposition as sc
import numpy as np
import matplotlib.ticker as mticker
import joblib
import torch
import torch.nn as nn
import random
from scipy.optimize import nnls
EPSILON = np.finfo(np.float32).eps


class UnsuperLayer(nn.Module):
    """
    Multiplicative update with Frobenius norm
    This can fit L1, L2 regularization.
    """

    def __init__(self, comp, features, l_1, l_2):
        super(UnsuperLayer, self).__init__()
        # an affine operation: y = Wx +b
        self.l_1 = l_1
        self.l_2 = l_2
        self.fc1 = nn.Linear(comp, comp, bias=False)
        self.fc2 = nn.Linear(features, comp, bias=False)

    def forward(self, y, x):
        denominator = torch.add(self.fc1(y), self.l_2 * y + self.l_1 + EPSILON)
        numerator = self.fc2(x)
        delta = torch.div(numerator, denominator)
        return torch.mul(delta, y)


class UnsuperNet(nn.Module):
    def __init__(self, n_layers, comp, features, n_classes, l_1=0, l_2=0, n_comp=30):
        super(UnsuperNet, self).__init__()
        self.n_comp = n_comp
        # Camadas existentes
        self.deep_nmfs = nn.ModuleList(
            [UnsuperLayer(comp, features, l_1, l_2) for _ in range(n_layers)])
        # Camada classificadora adicional
        self.classifier = nn.Linear(comp, n_classes)

    def forward(self, x):
        h = torch.rand(x.shape[0], self.n_comp,
                       device=x.device, dtype=torch.float)
        for layer in self.deep_nmfs:
            h = layer(h, x)
        # Saída do classificador
        out = self.classifier(h)
        return out


def cost_tns(v, w, h, l_1=0, l_2=0):
    # util.cost_tns(data.v_train.tns,data.w.tns,data.h_train.tns)
    d = v - h.mm(w)
    return (
        0.5 * torch.pow(d, 2).sum()
        + l_1 * h.sum()
        + 0.5 * l_2 * torch.pow(h, 2).sum()
    )/h.shape[0]


def tensoring(X):
    # conver numpy array to torch tensor
    return torch.from_numpy(X).float()


def build_data(V, H):

    V_tns = torch.from_numpy(V.T).float()
    H_tns = torch.from_numpy(H.T).float()

    return V_tns, H_tns


def train_supervised(V_tns, labels_tns, W_init_tns, num_layers, network_train_iterations, n_components, features, n_classes, lr=0.0005, l_1=0, l_2=0, verbose=False):
    """
    Train the supervised DeepNMF model with a classification layer.

    Parameters:
    - V_tns: Training dataset (features) in tensor format.
    - labels_tns: Class labels for the training dataset in tensor format.
    - W_init_tns: Initial weights for W (tensor format).
    - num_layers: Number of layers in the DeepNMF network.
    - network_train_iterations: Number of iterations for training the network.
    - n_components: Number of components for factorization.
    - features: Original features length for each sample vector.
    - n_classes: Number of target classes for classification.
    - lr: Learning rate for the optimizer.
    - l_1, l_2: Regularization parameters.
    - verbose: If True, print loss information during training.

    Returns:
    - model: Trained DeepNMF model with a classification layer.
    - train_cost: List of training costs.
    """

    # Initialize the model with a classification layer
    model = UnsuperNet(num_layers, n_components, features, n_classes, l_1, l_2)

    # Initialize parameters
    dnmf_w = W_init_tns.clone()
    for w in model.parameters():
        w.data.fill_(0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_cost = []

    for epoch in range(network_train_iterations):
        # Forward pass
        outputs = model(V_tns)
        loss = criterion(outputs, labels_tns)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{network_train_iterations}], Loss: {loss.item()}')

        train_cost.append(loss.item())

    print('Training completed.')
    return model, train_cost


def read_and_process_csv(file_path):
    # Carrega o dataset
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # pula a primeira linha

    # Divide cada linha em texto e classe
    texts, classes = zip(*[line.rsplit(',', 1) for line in lines])

    # Converte as classes para números
    le = LabelEncoder()
    classes = le.fit_transform(classes)

    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    V = vectorizer.fit_transform(texts).toarray()

    return V, classes


def label_documents(classes, n_labeled):

    # Escolhe uma classe aleatória para ser a positiva, e transforma o restante em negativa
    positive_class = classes[0]
    # positive_class = choice(np.unique(classes))
    classes = np.where(classes == positive_class, 1, 0)

    # Rotula uma quantidade n_labeled de documentos da classe positiva
    positive_indexes = np.where(classes == 1)[0]
    labeled = positive_indexes[:n_labeled]
    unlabeled = positive_indexes[n_labeled:]

    # Cria máscaras booleanas para os documentos rotulados e não rotulados
    labeled_mask = np.zeros(len(classes), dtype=bool)
    labeled_mask[labeled] = True
    unlabeled_mask = ~labeled_mask

    return labeled_mask, unlabeled_mask, labeled, positive_class


def predict_classes(reconstructed_V, threshold=0.5):
    # Placeholder for a real implementation
    predictions = np.sum(reconstructed_V, axis=0) > threshold
    return predictions


def main():
    # Caminho para o arquivo CSV
    file_path = 'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/Deep NMF/Datasets/tr41.mat.csv'
    V, classes = read_and_process_csv(file_path)

    n_samples, n_features = V.shape
    n_components = 30  # Ajuste conforme necessário
    n_classes = len(np.unique(classes))  # Número de classes únicas

    # Define o número de documentos rotulados da classe positiva
    n_labeled = 30

    # Rotula uma quantidade específica de documentos como positivos, o restante é não rotulado/negativo
    labeled_mask, unlabeled_mask, labeled, positive_class = label_documents(
        classes, n_labeled)

    # Ajusta os rótulos para serem compatíveis com a função de perda
    # Transforma rótulos em binário com a classe positiva sendo 1 e todas as outras sendo 0
    classes_binary = (classes == positive_class).astype(int)
    classes_tns = torch.tensor(
        classes_binary, dtype=torch.long)  # Para CrossEntropyLoss

    # Converte V para tensor
    V_tns = torch.tensor(V.T, dtype=torch.float)

    # Inicializa W e H com valores aleatórios
    W_init = np.random.rand(n_features, n_components)
    W_init_tns = torch.tensor(W_init.T, dtype=torch.float)

    # Parâmetros para o treinamento
    num_layers = 5
    network_train_iterations = 100
    lr = 0.001
    l_1 = 0.1
    l_2 = 0.1

    # Treina o modelo
    model, train_cost = train_supervised(
        V_tns, classes_tns, W_init_tns, num_layers, network_train_iterations,
        n_components, n_features, 2, lr, l_1, l_2, verbose=True)  # Assume 2 classes (positiva e negativa)

    # Avalia o desempenho do modelo usando F1 Score
    model.eval()  # Coloca o modelo em modo de avaliação
    with torch.no_grad():
        outputs = model(V_tns[unlabeled_mask])
        _, predicted = torch.max(outputs.data, 1)
        f1 = f1_score(classes_binary[unlabeled_mask],
                      predicted.numpy(), average='binary')

    print(f'F1 Score of the model on the unlabeled documents: {f1}')


if __name__ == "__main__":
    main()
