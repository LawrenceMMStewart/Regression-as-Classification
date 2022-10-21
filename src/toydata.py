import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
tf.config.set_visible_devices([], device_type='GPU')


def relu(x):
    """
    returns the ReLu of an input
    array x
    """
    r = np.zeros(x.shape)
    r[x > 0] = x[x > 0]
    return r

def onehot(x, C):
    """
    converts a column or row vector
    with integer values into one-hot form
    """
    x = x.flatten()
    o = np.eye(C)[x]
    return o

class triangle():
    def __init__(self, scale=1, a=0, b=1):
        """
        triangle wave on range [a,b]

        Args:

        x : np.array
        scale : float (magnitude of wave)
        a : float (domain lower bound)
        b : float (domain upper bound)

        Returns:

        y: np.array
        """

        self.scale = scale
        assert b > a
        self.a = a
        self.b = b

    def __call__(self, x):

        a = self.a
        b = self.b
        scale = self.scale
        mid = (a + b) / 2
        diff = (b - a) / 2
        mask1 = x >= a
        mask2 = x <= b
        distance = np.abs(x - mid)
        y = 1 - (distance / diff)
        return y * scale * mask1 * mask2


def multiscale_triangle(x):
    """
    Multiscale triangle wave
    data (in the limit of the MLP function class)
    """
    triangles = []
    # first large triangle between -1 and 0
    triangles.append(triangle(scale=1, a=-1, b=0))
    # second large triangle between 0 and 1
    triangles.append(triangle(scale=1, a=0, b=1))

    # multiscale triangle detail in the triangles
    triangles.append(triangle(scale=-0.3, a=-0.7, b=-0.5))
    triangles.append(triangle(scale=-0.4, a=0.55, b=0.75))

    Y = [f(x) for f in triangles]
    return sum(Y)




def get_bin_domains(fn, bins):
    """
    given a function on [-1,1] and C bins
    returns a list of length C where each
    list contains of a list of of x-axis intervals
    denoting where the function in that respective bin
    """
    C = len(bins) - 1
    x = np.linspace(-1, 1, 100000).astype('float32')
    y = fn(x)
    yt = np.digitize(y, bins) - 1

    subdomains = [[ ] for _ in range(C)]
    subdomains[yt[0]].append(-1)
    for i in range(1, len(x) - 1):
        if yt[i] == yt[i - 1]:
            pass
        else:
            subdomains[yt[i - 1]].append(x[i - 1])
            subdomains[yt[i]].append(x[i])

    subdomains[yt[-1]].append(1)
    subdomains = [np.array(m) for m in subdomains]

    # at this point each array in subdomains will have 2k elements
    # split into a list of k intervals
    intervals = []
    for inter in subdomains:
        l = np.split(inter, len(inter) // 2)
        l = [tuple(v) for v in l]
        intervals.append(l)
    return intervals


def balanced_xgen(fn, n, bins, rstate):
    C = len(bins) - 1
    assert n % C == 0, "n must be a multiple of C for balanced data"
    nc = n // C

    intervals = get_bin_domains(fn, bins)
    xvals = []
    for dom in intervals:
        xcur = []
        for (a,b) in dom:
            xcur += rstate.uniform(a, b, size=(nc)).tolist()
        xvals += rstate.choice(xcur, nc).tolist()

    return np.sort(np.array(xvals)).reshape(-1, 1).astype('float32')


def balanced_datagen_reg(fn, n, bins, rstate):
    x = balanced_xgen(fn, n, bins, rstate)
    y = fn(x)
    x = np.concatenate((x, np.ones_like(x)), axis=-1)
    return x, y


def balanced_datagen_class(fn, n, bins, rstate):
    x = balanced_xgen(fn, n, bins, rstate)
    y = fn(x)
    x = np.concatenate((x, np.ones_like(x)), axis=-1)
    yc = np.digitize(y, bins) - 1
    return x, y, yc

def truthgen_reg(fn):
    x = np.linspace(-1, 1, 1000).reshape(-1, 1).astype('float32')
    y = fn(x)
    return np.concatenate((x, np.ones_like(x)), axis=1), y

class BalancedDataset():
    def __init__(self, fn, nbs, bs, ne, bins, rstate, reg=True):
        """
        Sample train data uniformly (n = bs  * nbs) with equal number of points in each bin.
        Sample eval data (ne points) uniformly.
        """
        if reg:
            # train
            x, y = balanced_datagen_reg(fn, nbs * bs, bins, rstate)
            # eval
            xe, ye = balanced_datagen_reg(fn, ne, bins, rstate)
        else:
            # train  (yc = discretized label)
            x, y, yc = balanced_datagen_class(fn, nbs * bs, bins, rstate)
            # eval
            xe, ye, yce  = balanced_datagen_class(fn, ne, bins, rstate)

        # truth data for the whole function
        self.xt, self.yt = truthgen_reg(fn)

        self.fn = fn
        self.reg = reg
        self.x, self.y = x, y
        self.xe, self.ye = xe, ye
        self.bs, self.nbs = bs, nbs
        self.n, self.ne = nbs * bs, ne
        self.rstate = rstate

        if not self.reg:
            self.yc, self.ych = yc, onehot(yc, len(bins) - 1)
            self.yce, self.yche = yce, onehot(yce, len(bins) - 1)

    def batch_and_shuffle(self):
        perm = self.rstate.permutation(np.arange(self.x.shape[0]))
        for i in range(self.nbs):
            ids = perm[self.bs * i : self.bs * (i + 1)]
            if self.reg:
                yield (self.x[ids], self.y[ids])
            else:
                yield (self.x[ids], self.y[ids], self.yc[ids], self.ych[ids])

def plot_opt(params):
    """
    plot the optimal lines
    for the MLP
    t(w * a[0], w * a[1]) : t>0
    """
    w, a = params

    p = a * w.flatten()
    output = []
    for i in range(p.shape[1]):
        p_ = p[:, i]
        vals = np.array([[0, 0], p_ * (1.), p_ * (1000)])
        a, = plt.plot(
            vals[:, 0],
            vals[:, 1],
            linestyle='--',
            alpha=0.8,
            label='opt',
            color='g')
        output.append(a)
    return output


def weight_pos(params):
    """
    gets the weights positions
    in 2D space:

    abs(w)*a[0], abs(w)*a[1]
    """
    w, a = params
    return abs(w.flatten()) * a

    return r



