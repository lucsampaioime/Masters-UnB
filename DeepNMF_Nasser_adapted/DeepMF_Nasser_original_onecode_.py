from matplotlib import pyplot as plt
from matplotlib.pyplot import text
import pandas as pd
import sklearn.decomposition as sc
import numpy as np
import matplotlib.ticker as mticker
import joblib


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


@dataclass
class Matensor:
    mat: np.ndarray
    tns: torch.Tensor


@dataclass
class Alldata:
    v_train: Matensor
    v_test: Matensor
    v_test: Matensor
    h_train: Matensor
    h_test: Matensor
    h_0_train: Matensor
    h_0_test: Matensor
    w: Matensor
    w_init: Matensor


def tensoring(X):
    # conver numpy array to torch tensor
    return torch.from_numpy(X).float()


def build_data(V, W, H, TRAIN_SIZE=0.80, index=None):
    # create a dataclass object that includes all necessary datasets to train a model
    n_components = H.shape[0]
    features, samples = V.shape
    # split train/test
    TRAIN_SIZE = 0.80
    if index is None:
        mask = np.random.rand(samples) < TRAIN_SIZE
    else:
        mask = index

    H_init = np.ones((n_components, samples))
    W_init = np.ones((features, n_components))/features

    data = Alldata(
        v_train=Matensor(V[:, mask], tensoring(V[:, mask].T)),
        v_test=Matensor(V[:, ~mask], tensoring(V[:, ~mask].T)),
        h_train=Matensor(H[:, mask], tensoring(H[:, mask].T)),
        h_test=Matensor(H[:, ~mask], tensoring(H[:, ~mask].T)),
        h_0_train=Matensor(H_init[:, mask], tensoring(H_init[:, mask].T)),
        h_0_test=Matensor(H_init[:, ~mask], tensoring(H_init[:, ~mask].T)),
        w=Matensor(W, tensoring(W.T)),
        w_init=Matensor(W_init, tensoring(W_init.T)),
    )
    return data, n_components, features, samples


def train_unsupervised(
    data: Alldata,
    num_layers,
    network_train_iterations,
    n_components,
    verbose=False,
    lr=0.0005,
    l_1=0,
    l_2=0,
    include_reg=True
):
    v_train = data.v_train.tns
    h_0_train = data.h_0_train.tns
    features = v_train.shape[1]
    # build the architicture
    if include_reg:
        deep_nmf = UnsuperNet(num_layers, n_components, features, l_1, l_2)
    else:
        deep_nmf = UnsuperNet(num_layers, n_components, features, 0, 0)
    # initialize parameters
    dnmf_w = data.w_init.tns
    for w in deep_nmf.parameters():
        w.data.fill_(0.1)

    optimizerADAM = optim.Adam(deep_nmf.parameters(), lr=lr)
    # Train the Network
    inputs = (h_0_train, v_train)
    test = (data.h_0_test.tns, data.v_test.tns)
    dnmf_train_cost = []
    dnmf_test_cost = []
    for i in range(network_train_iterations):
        out = deep_nmf(*inputs)
        test_out = deep_nmf(*test)

        loss = cost_tns(v_train, dnmf_w, out, l_1, l_2)

        if verbose:
            print(i, loss.item())

        optimizerADAM.zero_grad()
        loss.backward()
        optimizerADAM.step()

        # keep weights positive after gradient decent
        for w in deep_nmf.parameters():
            w.data = w.clamp(min=0, max=inf)
        h_out = torch.transpose(out.data, 0, 1)
        h_out_t = out.data

        # NNLS
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

        w_arrays = [nnls(out.data.numpy(), data.v_train.mat[f])[0]
                    for f in range(features)]
        nnls_w = np.stack(w_arrays, axis=-1)
        dnmf_w = torch.from_numpy(nnls_w).float()

        dnmf_train_cost.append(loss.item())

        # test performance
        dnmf_test_cost.append(
            cost_tns(data.v_test.tns, dnmf_w, test_out, l_1, l_2).item())

    return deep_nmf, dnmf_train_cost, dnmf_test_cost, dnmf_w
