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


def build_data(V, H):
    """
    Prepare the data for NMF model without splitting into training and test sets.

    Parameters:
    - V: Term-document matrix (numpy array).
    - H: Initial H matrix (numpy array).

    Returns:
    - V_tns: Term-document matrix as a torch tensor.
    - H_tns: Initial H matrix as a torch tensor.
    """
    # Convert numpy arrays to torch tensors
    V_tns = torch.from_numpy(V.T).float()
    H_tns = torch.from_numpy(H.T).float()

    return V_tns, H_tns


def train_unsupervised(V, W_init, H_init, num_iterations, lr=0.0005, l_1=0, l_2=0, positive_class_indices=None, verbose=False):
    """
    Train the unsupervised NMF model with specified initializations and parameters, including a step to emphasize
    documents belonging to the positive class by setting their corresponding values in H to the highest value observed.

    Parameters:
    - V: Input data matrix (torch.Tensor).
    - W_init: Initial value for W (torch.Tensor).
    - H_init: Initial value for H (torch.Tensor).
    - num_iterations: Number of iterations to run the training.
    - lr: Learning rate for the optimizer.
    - l_1: L1 regularization parameter.
    - l_2: L2 regularization parameter.
    - positive_class_indices: Indices of documents in the positive class.
    - verbose: If True, print cost at each iteration.

    Returns:
    - W: Updated W matrix.
    - H: Updated H matrix.
    - cost_history: List of costs per iteration.
    """
    V = V.T.float()
    W = W_init.T.float()
    H = H_init.T.float()

    cost_history = []

    print(f"Initial shapes - V: {V.shape}, W: {W.shape}, H: {H.shape}")

    for i in range(num_iterations):
        # Update H using a gradient descent step
        H.requires_grad_(True)
        optimizerADAM = torch.optim.Adam([H], lr=lr)

        optimizerADAM.zero_grad()
        reconstruction = W @ H
        if verbose:
            print(
                f"Iteration {i} - W: {W.shape}, H: {H.shape}, Reconstruction: {reconstruction.shape}")
        cost = torch.norm(V - reconstruction, 'fro') + l_1 * \
            torch.norm(H, 1) + l_2 * torch.norm(H, 2)
        cost.backward()
        optimizerADAM.step()

        # After updating H, set the highest value for the positive class if specified
        if positive_class_indices is not None:
            highest_value = H.data.max().item()  # Find the highest value in H
            for idx in positive_class_indices:
                # Set highest value for the positive class
                H.data[:, idx] = highest_value

        # Clamp H to ensure non-negativity
        H.data = torch.clamp(H.data, min=0)

        # Update W using NNLS for each column of V
        for column in range(V.shape[1]):
            W[:, column] = torch.from_numpy(
                nnls(H.detach().numpy().T, V[:, column].numpy())[0]).float()

        cost_history.append(cost.item())

        if verbose:
            print(f"Iteration {i+1}/{num_iterations}, Cost: {cost.item()}")

        if i % 10 == 0 and verbose:  # Every 10 iterations, adjust as needed
            print(
                f"Iteration {i} - Loss: {loss.item()}, W shape: {W.shape}, H shape: {H.shape}")

    return W, H, cost_history


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

    n_labeled = 10  # Or get this from user input.
    labeled_mask, unlabeled_mask, labeled, positive_class = label_documents(
        classes, n_labeled)

    print("Positive Class Selected:", positive_class)

    n_samples, n_features = V.shape
    n_components = 20  # You can define this based on your needs

    V = V.transpose()

    # Initialize W and H with random values
    W = np.random.rand(n_features, n_components)
    H = np.random.rand(n_components, n_samples)

    # Prepare data (convert to tensors without splitting)
    V_tns, H_tns = build_data(V, H)

    # Convert W to tensor
    W_init_tns = torch.from_numpy(W.T).float()

    # Parameters for the training
    num_layers = 5
    network_train_iterations = 100
    lr = 0.001
    l_1 = 0.1
    l_2 = 0.1

    # Train the model
    model, train_cost, dnmf_w = train_unsupervised(
        V_tns, W_init_tns, H_tns, network_train_iterations, lr, l_1, l_2, labeled, verbose=True)

    # Calculate error
    reconstructed_V = H_tns.mm(dnmf_w)
    error = torch.norm(V_tns - reconstructed_V, p='fro').item()
    print("Reconstruction Error:", error)

    # Reverse transpose for compatibility with classification metrics calculations
    V = V.transpose()
    W_x = H.transpose()
    H_x = W.transpose()

    # Determine the most representative topic for the positive class
    positive_class_index = np.argmax(np.sum(W_x[classes == 1], axis=0))
    # Calculate classification metrics
    preds = np.argmax(W_x, axis=1) == positive_class_index
    preds = preds.astype(int)

    accuracy = accuracy_score(classes[unlabeled_mask], preds[unlabeled_mask])
    precision = precision_score(
        classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
    recall = recall_score(
        classes[unlabeled_mask], preds[unlabeled_mask], average='weighted', zero_division=0)
    f1 = f1_score(classes[unlabeled_mask], preds[unlabeled_mask],
                  average='weighted', zero_division=0)

    print(
        f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


if __name__ == "__main__":
    main()
