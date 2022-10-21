from utils import get_bins, get_midpoints
from math import ceil, sqrt
from model import MLP_Mean_Field_Init, batched_forward, CEupdate, L2update, L2, CE, relu
from toydata import onehot
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow.config as tfconfig
from jax.nn import softmax
from jax import jit, grad
import jax.numpy as jnp
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
tfconfig.set_visible_devices([], device_type='GPU')

def train_regression(params, X, Y, lr=0.05, max_epochs=10000):

    for i in tqdm(range(max_epochs)):
        params= L2update(params, X, Y, lr)
        marker = max_epochs // 100
        if i % marker == 0:
            loss = L2(params, X, Y)
            print(loss)
    return params

def train_classification(params, X, Ych, lr=0.05, max_epochs=100000):
    for i in tqdm(range(max_epochs)):
        params = CEupdate(params, X, Ych, lr)
        marker = max_epochs // 100
        if i % marker == 0:
            print(CE(params, X, Ych))
    return params

def plotpreds(P, X, Y, bins=None, angle=(20,-120)):

    preds = batched_forward(P, X)

    if bins is not None:
        midpoints = get_midpoints(bins)
        midpoints = midpoints.reshape(-1, 1)
        probs = softmax(preds, axis=-1)
        preds = probs @ midpoints

    plt.style.use('seaborn-colorblind')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    fig = plt.figure(figsize=(9.6, 7.2))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], preds[:, 0], label='model predictions')
    ax2.scatter(X[:, 0], X[:, 1], Y[:, 0], label=r'$\mu_T$')
    ax1.legend(loc='upper right', prop={'size': 18})
    ax2.legend(loc='upper right', prop={'size': 18})
    ax1.set_zticks([])
    ax2.set_zticks([])
    ax1.view_init(angle[0], angle[1])
    ax2.view_init(angle[0], angle[1])
    plt.show()
    plt.close()

def threshold(P, X, percent=0.01, bins=None, angle=(20, -120)):
    """
    plots the predictions of a model after discarding all features
    whose weights |a_j||b_j| are not greater than `percent` %  of the model's
    total weight sum_j |a_j||b_j| .

    This allows visual inspection as to whether the threshold has got rid of
    trivial non used features or is too strong and affecting the prediction
    """

    try:
        Pt = relevant_features(P, percent=percent)
    except:
        print("thresholding too strong, no features satisfied conditions")
        return None
    preds = batched_forward(P, X)


    if bins is not None:
        midpoints = get_midpoints(bins)
        midpoints = midpoints.reshape(-1, 1)
        probs = softmax(preds, axis=-1)
        preds = probs @ midpoints

    preds2 = batched_forward(Pt, X)

    if bins is not None:
        probs2 = softmax(preds2, axis=-1)
        preds2 = probs @ midpoints

    plt.style.use('seaborn-colorblind')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    fig = plt.figure(figsize=(9.6, 7.2))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], preds[:, 0], label='params')
    ax2.scatter(X[:, 0], X[:, 1], preds2[:, 0], label=fr'$\geq{(percent * 100):.4f}\%$ weight')
    ax1.legend(loc='upper right', prop={'size': 18})
    ax2.legend(loc='upper right', prop={'size': 18})
    ax1.view_init(angle[0], angle[1])
    ax2.view_init(angle[0], angle[1])
    ax1.set_zticks([])
    ax2.set_zticks([])
    plt.show()
    plt.close()

def get_active(P):
    """
    returns the ids of the features whose
    critical lines cross the unit square
    """

    A, B = P
    M = A.shape[1]
    ac_ids = []

    xcorns = np.array([-1, -1, 1, 1])
    ycorns = np.array([-1, 1, -1, 1])
    for j in range(M):
        a1, a2, a3 = A[:, j]
        l = lambda x : (- a2 / a1) * x - (a3 / a1)
        if len(np.unique(l(xcorns) >= ycorns)) > 1:
            ac_ids.append(j)

    return ac_ids

def get_significant(P, percent=0.01):
    """
    returns the ids of the weights which are
    at least percent % of the total weight
    """
    A, B = P
    NA = np.linalg.norm(A, axis=0)
    NB = np.linalg.norm(B, axis=1)

    T = np.sum(NA * NB)

    return np.where(NB >= T * percent)[0]

def get_ids(P, percent=0.01):
    """
    removes features that are affine and dont intersect
    with the unit grid, as well as features whose weight
    |a_j||b_j| does not correspond to at least of percent
    % of the total weight on the model.
    """
    ac_ids = get_active(P)
    sig_ids = get_significant(P, percent)
    return [i for i in sig_ids if i in ac_ids]

def sortbyweight(P):
    """
    sorts the parameters of the model by their weight
    """
    A, B = P
    NA = np.linalg.norm(A, axis=0)
    NB = np.linalg.norm(B, axis=1)
    weight = NA * NB
    order = np.argsort(-weight)
    A = A[:, order]
    B = B[order, :]
    return (A, B)


def relevant_features(P, percent=0.01):
    """
    gets the features which are relevant to the model's
    prediction, discarding those that do not matter,
    as described in the appendix.
    """
    ids = get_ids(P, percent)
    assert len(ids) > 0 , "threshold too powerful"
    Prelevant_features = (P[0][:, ids], P[1][ids, :])
    return sortbyweight(Prelevant_features)

def plotfeats(Pord, x, angle=(-20, 120)):
    A, B = Pord
    M = A.shape[1]
    K = ceil(sqrt(M))
    K = min(K, 5)
    plt.style.use('seaborn-colorblind')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    fig = plt.figure()
    for j in range(min(K ** 2, A.shape[1])):
        ax = fig.add_subplot(K, K, j+1,  projection='3d')
        aj = A[:, j]
        title = np.linalg.norm(aj) * np.linalg.norm(B[j])
        if B.shape[1] > 1:
            yt = np.linalg.norm(B[j]) * relu(x @ aj)
        else:
            yt = B[j] * relu(x @ aj)

        ax.scatter(x[:, 0], x[:, 1], yt, alpha=0.3, marker='.', s=8, label=f'{j}')
        ax.view_init(angle[0], angle[1])
        ax.set_title(f"{title:.4f}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

    plt.show()
    plt.close()


def plotlines(P, title=None):
    x = np.linspace(-1, 1, 50)
    A, B = P
    M = A.shape[1]
    plt.style.use('seaborn-colorblind')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.figure(figsize=(9.6, 7.2))

    for j in range(M):
        a1, a2, a3 = A[:, j]
        l = lambda x : (- a2 / a1) * x - (a3 / a1)
        y = l(x)
        plt.plot(x, y, alpha=0.9)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xticks(size=22)
    plt.yticks(size=22)
    plt.xlabel(r'$x_1$', size=28)
    plt.ylabel(r'$x_2$', size=28)
    # plt.title(title)
    plt.show()
    plt.close()




if __name__ == "__main__":
    import jax.random as jrandom
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    seed = 7421
    key = jrandom.PRNGKey(seed)
    np.random.seed(seed)

    v = np.linspace(-1, 1, 25)
    X1, X2 = np.meshgrid(v, v)
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    x = np.concatenate((X1, X2, np.ones_like(X1)), axis=1)

    # Pdat = parameters of teacher model
    Pdat = MLP_Mean_Field_Init([3,3,1], key)

    C = 25
    y = batched_forward(Pdat, x)
    cbins, obins = get_bins(C, y.max(), y.min())
    yc = np.digitize(y, cbins) - 1
    ych = onehot(yc, C)

    # parameters of regression model
    PR = MLP_Mean_Field_Init([3, 500, 1], key)
    # parameters of classification model
    PC = MLP_Mean_Field_Init([3, 500, C], key)
    PR = train_regression(PR, x, y, lr=0.1, max_epochs=50000)
    PC = train_classification(PC, x, ych, lr=0.1, max_epochs=10000)

    plotpreds(PR, x, y)
    threshold(PR, x, 0.04)
    plotlines(relevant_features(PR, 0.04), "regression")

    plotpreds(PC,x,y,cbins)
    threshold(PC, x, percent=0.003, bins=cbins)
    plotlines(relevant_features(PC, 0.003), "classification")

