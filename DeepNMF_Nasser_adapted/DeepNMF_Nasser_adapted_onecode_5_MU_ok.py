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
    """
    Class for a Regularized DNMF with varying layers number.
    Input:
        -n_layers = number of layers to construct the Net
        -comp = number of components for factorization
        -features = original features length for each sample vector(mutational sites)
    each layer is MU of Frobenius norm
    """

    def __init__(self, n_layers, comp, features, l_1=0, l_2=0):
        super(UnsuperNet, self).__init__()
        self.n_layers = n_layers
        self.deep_nmfs = nn.ModuleList(
            [UnsuperLayer(comp, features, l_1, l_2)
             for i in range(self.n_layers)]
        )

    def forward(self, h, x):
        # sequencing the layers and forward pass through the network
        for i, l in enumerate(self.deep_nmfs):
            h = l(h, x)
        return h


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


def multiplicative_update(V, W, H, num_iterations):
    for i in range(num_iterations):
        H_update = (W.T @ V) / (W.T @ W @ H + EPSILON)
        H *= H_update

        W_update = (V @ H.T) / (W @ H @ H.T + EPSILON)
        W *= W_update

    return W, H


def build_data(V, W, H, TRAIN_SIZE=0.80, index=None):
    # Split the data into training and testing sets
    samples = V.shape[1]
    if index is None:
        mask = np.random.rand(samples) < TRAIN_SIZE
    else:
        mask = index

    V_train = V[:, mask]
    V_test = V[:, ~mask]
    H_train = H[:, mask]
    H_test = H[:, ~mask]

    # Convert numpy arrays to torch tensors
    V_train_tns = torch.from_numpy(V_train.T).float()
    V_test_tns = torch.from_numpy(V_test.T).float()
    H_train_tns = torch.from_numpy(H_train.T).float()
    H_test_tns = torch.from_numpy(H_test.T).float()

    return V_train, V_test, H_train, H_test, V_train_tns, V_test_tns, H_train_tns, H_test_tns


def train_unsupervised(V_train_tns, V_test_tns, H_train_tns, H_test_tns, W_init_tns, num_layers, network_train_iterations, n_components, features, lr=0.0005, l_1=0, l_2=0, verbose=False, include_reg=True):
    """
    Train the unsupervised deep NMF model.

    Parameters:
    - V_train_tns, V_test_tns: Training and testing sets of V (tensor format).
    - H_train_tns, H_test_tns: Training and testing sets of H (tensor format).
    - W_init_tns: Initial weights for W (tensor format).
    - num_layers: Number of layers in the deep NMF network.
    - network_train_iterations: Number of iterations for training the network.
    - n_components: Number of components for factorization.
    - features: Original features length for each sample vector.
    - lr: Learning rate for the optimizer.
    - l_1, l_2: Regularization parameters.
    - verbose: If True, print loss information during training.
    - include_reg: Include regularization in the model if True.

    Returns:
    - deep_nmf: Trained deep NMF model.
    - dnmf_train_cost: List of training costs.
    - dnmf_test_cost: List of testing costs.
    - dnmf_w: Trained weights for W.
    """

    # Build the architecture
    deep_nmf = UnsuperNet(num_layers, n_components, features, l_1, l_2) if include_reg else UnsuperNet(
        num_layers, n_components, features, 0, 0)

    # Initialize parameters
    dnmf_w = W_init_tns.clone()
    for w in deep_nmf.parameters():
        w.data.fill_(0.1)

    optimizerADAM = torch.optim.Adam(deep_nmf.parameters(), lr=lr)

    # Train the Network
    inputs = (H_train_tns, V_train_tns)
    test = (H_test_tns, V_test_tns)
    dnmf_train_cost = []
    dnmf_test_cost = []

    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        test_out = deep_nmf(*test)
        loss = cost_tns(V_train_tns, dnmf_w, out, l_1, l_2)

        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # Keep weights positive after gradient descent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0)

        h_out = torch.transpose(out.data, 0, 1)

        # NNLS
        w_arrays = [nnls(out.data.numpy(), V_train_tns[:, f].numpy())[0]
                    for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

        dnmf_train_cost.append(loss.item())

        # Test performance
        dnmf_test_cost.append(
            cost_tns(V_test_tns, dnmf_w, test_out, l_1, l_2).item())

    return deep_nmf, dnmf_train_cost, dnmf_test_cost, dnmf_w


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
    # Path to your CSV file
    file_path = 'C:/Users/lucsa/Dropbox/Data Science/Mestrado UNB/Dissertação/Experimentos/Testes/Deep NMF/Datasets/CSTR.csv'
    V, classes = read_and_process_csv(file_path)

    # Ensuring the number of documents matches the number of class labels

    n_labeled = 30  # Or get this from user input
    labeled_mask, unlabeled_mask, labeled, positive_class = label_documents(
        classes, n_labeled)

    print("Positive Class Selected:", positive_class)

    n_samples, n_features = V.shape
    n_components = 20  # You can define this based on your needs
    print("Shape of V after processing CSV:", V.shape)
    print(n_features)
    print(n_samples)

    V = V.transpose()
    print("Shape of V after transpose:", V.shape)

    # Initialize W and H with random values
    W = np.random.rand(n_features, n_components)
    H = np.random.rand(n_components, n_samples)
    print("Shape of W:", W.shape)
    print("Shape of H:", H.shape)

    # Perform Multiplicative Update
    mu_num_iterations = 500  # Define the number of iterations
    W, H = multiplicative_update(V, W, H, mu_num_iterations)

    # Prepare data (split and convert to tensors)
    V_train, V_test, H_train, H_test, V_train_tns, V_test_tns, H_train_tns, H_test_tns = build_data(
        V, W, H)

    # Convert W to tensor
    W_init_tns = torch.from_numpy(W.T).float()

    # Parameters for the training
    num_layers = 5
    network_train_iterations = 100
    lr = 0.001
    l_1 = 0.1
    l_2 = 0.1

    # Train the model
    model, train_cost, test_cost, dnmf_w = train_unsupervised(
        V_train_tns, V_test_tns, H_train_tns, H_test_tns, W_init_tns, num_layers, network_train_iterations, n_components, n_features, lr, l_1, l_2, verbose=True)

    # Calculate error
    reconstructed_V = H_train_tns.mm(dnmf_w)
    error = torch.norm(V_train_tns - reconstructed_V, p='fro').item()
    print("Reconstruction Error:", error)

    V = V.transpose()
    print("Shape of V after transpose 2:", V.shape)

    W_x = H.transpose()
    print("Shape of W after transpose 2:", W_x.shape)
    H_x = W.transpose()
    print("Shape of H after transpose 2:", H_x.shape)

    # Determina qual tópico é o mais representativo para a classe positiva [usar uma das opções abaixo]
    positive_class_index = np.argmax(np.sum(W_x[classes == 1], axis=0))
    # positive_class_index = 0

    # Calcula as métricas de classificação
    preds = np.argmax(W_x, axis=1) == positive_class_index
    preds = preds.astype(int)

    accuracy = accuracy_score(classes[unlabeled_mask], preds[unlabeled_mask])
    precision = precision_score(
        classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
    recall = recall_score(
        classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
    f1 = f1_score(classes[unlabeled_mask],
                  preds[unlabeled_mask], average='weighted', zero_division=0)

    print(
        f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


if __name__ == "__main__":
    main()
