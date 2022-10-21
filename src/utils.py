import numpy as np
import matplotlib.pyplot as plt
import string
import random
import jax.numpy as jnp
from model import batched_forward
import os
from jax.nn import softmax


def ema(x, n):
    """
    compute an n period exponential
    moving average on a sequence x
    """
    x = np.asarray(x)
    weights = np.exp(np.linspace(-1., 0., n))
    weights /= weights.sum()
    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


def generate_exp_name():
    allowed_chars = string.ascii_letters + string.digits
    exp_name = ''.join(random.choice(allowed_chars) for x in range(30))
    return exp_name

def get_bins(C, max_val=1, min_val=0):
    """
    Produce the bin edges for cutting the interval
    [min_val, max_val] into C bins where the first bin
    has midpoint min_val and the last bin has midpoint
    max_val. The function also returns open bins for
    the option of noisy data.
    """

    d = max_val - min_val
    # b is the length of a bin
    b = d / (C - 1)
    a0 = min_val - (b / 2)
    bins = []
    for i in range(C + 1):
        bins.append(a0 + b * i)
    open_bins = [b for b in bins]
    open_bins[0] = -np.inf
    open_bins[-1] = np.inf
    return bins, open_bins



def get_midpoints(bins):
    # calculate the midpoints of the bins
    values = [(bins[i] + bins[i + 1]) / 2 for i in range(0, len(bins) - 1)]
    values = np.array(values)
    return values


def plotlosses(losses, args, title='', yscale='log'):

    # plt.style.use('ggplot')
    plt.style.use('seaborn-colorblind')
    fig = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    l = len(losses)
    xs = [i *  args['tepochs'] / l for i in range(l)]
    plt.plot(xs, losses, alpha=0.2, label='raw')
    # plt.plot(xs, losses, color="#D55E00", alpha=0.2, label='raw')
    # plt.plot(xs, ema(losses, min(5 * l // 100, 1000)), alpha=0.8, label='smoothing')
    plt.plot(xs, ema(losses, min(5 * l // 100, 1000)), alpha=0.8, label='smoothing')
    # plt.ylabel(title)
    plt.yscale(yscale)
    plt.xlabel('Epoch', size=20)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(loc='upper right', prop={'size': 20})
    if args['save']:
        plt.savefig(
            os.path.join(args['save_dir'],
                         title+".pdf")
        )
    else:
        plt.show()

    plt.close()



def plot_gradtraj(args, grad_traj):
    """
    plots eigevalues of covariance matrix
    of the first 1000 gradients obtained during
    training
    """
    # eigenvalues of AA^T is the same as A^TA
    cov = grad_traj @ grad_traj.T
    eigs = np.linalg.eigvalsh(cov + 1e-7)

    # plt.style.use('ggplot')
    plt.style.use('seaborn-colorblind')
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    eigs = np.flip(np.sort(eigs))

    top5 = np.flip(np.sort(eigs))[:50]
    xs = np.arange(len(top5)) + 1
    plt.plot(xs, top5, color="#D55E00", alpha=0.4)
    plt.scatter(xs, top5, color="#D55E00", alpha=0.8, marker='.')
    plt.yscale('log')
    plt.ylabel(r'$\lambda$', size=20)
    # plt.xlabel('')
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    if args['save']:
        plt.savefig(
            os.path.join(args['save_dir'],
                         f"eigs.pdf")
        )
    else:
        plt.show()
    plt.close()


def grad_angles(args, grad_traj):
    """
    plots angles between first 1000
    gradients in training
    """
    # plt.style.use('ggplot')
    plt.style.use('seaborn-colorblind')
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    norm = np.linalg.norm(grad_traj, axis=1).reshape(-1, 1)
    gtn = grad_traj / norm
    ncov = gtn @ gtn.T
    # clear up any floating point calculation error
    ncov = np.clip(ncov, a_min=-1, a_max=1)
    theta = np.arccos(ncov)
    # convert from radians to degrees
    theta *= 180 / np.pi
    plt.imshow(theta, cmap='cividis')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(label=r'$\theta$',weight='bold', size=22)
    plt.xticks([])
    plt.yticks([])
    if args['save']:
        plt.savefig(
            os.path.join(args['save_dir'],
                         "theta.pdf")
        )
    else:
        plt.show()
    plt.close()




def plotprediction(ds, params, args):
    """
    plots the learnt function
    """
    # plt.style.use('ggplot')
    plt.style.use('seaborn-colorblind')

    # forward pass
    logits = np.array(batched_forward(params, ds.xt))

    if args['C'] == 1:
        preds = logits
    else:
        midpoints = np.array(args['midpoints']).reshape(-1, 1)
        # convert classification to regression
        probs = softmax(logits, axis=-1)
        preds = np.array(probs @ midpoints)

    for i in range(2):
        if i == 0:
            X, Y = ds.x, ds.y
            label='train'
        else:
            X, Y = ds.xe, ds.ye
            label='eval'
        plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        # plt.plot(ds.xt[:, 0], ds.yt, color="#009E73", label=r'$f_{\mu_T}$', alpha=0.8)
        plt.plot(ds.xt[:, 0], ds.yt, label=r'$f_{\mu_T}$', alpha=0.6)
        # plt.plot(ds.xt[:, 0], preds, color="#D55E00", label='model prediction', alpha=0.8)
        plt.plot(ds.xt[:, 0], preds, label='model prediction')
        # plt.scatter(X[:, 0], Y, color="#0072B2", marker='x', s=8, label=label + ' data', zorder=3, alpha=0.9)
        plt.scatter(X[:, 0], Y, marker='x', s=8, label=label + ' data', zorder=3, alpha=0.9)
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
        plt.legend(loc='lower left', prop={'size': 20})
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)


        if args['save']:
            plt.savefig(
                os.path.join(args['save_dir'],
                             f"predictions_{label}.pdf")
            )
        else:
            plt.show()
        plt.close()


def process_args(args):
    """
    takes a namespace args
    and processes it for the experiment
    """
    args['layer_sizes']  = [int(l) for l in args['layer_sizes'].split(',')]
    args['C'] = args['layer_sizes'][-1]

    # if classification problem create bins
    cbins, obins = get_bins(args['dC'], max_val=1, min_val=0)
    args['cbins'] = cbins
    args['obins'] = obins
    args['midpoints'] = get_midpoints(cbins)

    # if saving the experiment, generate output folder
    if args['save']:
        if args['exp_name'] == '':
            args['exp_name'] = generate_exp_name()
        save_dir = os.path.join("experiments", args['exp_name'])
        if os.path.isdir(save_dir):
            print("save dir")
            pass
        else:
            os.mkdir(save_dir)
        args['save_dir'] = save_dir

    return args


def regression_support(x, y):
    """
    Calculates R_reg
    """
    x = x[:, 0].flatten().astype('float64')
    y = y.flatten().astype('float64')
    supp_ids = []
    n = len(x)
    for i in range(n):
        # include first point as part of support
        if i == 0:
            supp_ids.append(i)
        # include last point as part of support
        elif i == n - 1:
            supp_ids.append(i)
        else:
            # over flow problem:
            prev_grad = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            curr_grad = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            if np.isnan(curr_grad):
                curr_grad = (y[i + 2] - y[i]) / (x[i + 2] - x[i])
            if np.abs(prev_grad  - curr_grad) > 1e-3:
                supp_ids.append(i)
            else:
                pass
    return supp_ids


def classification_support(x, yc):
    """
    calculates R_class
    """
    x = x[:, 0].flatten().astype('float64')
    yc = yc.flatten()
    supp_ids = []
    n = len(x)
    for i in range(n):
        # include first point as part of support
        if i == 0:
            supp_ids.append(i)
        # include last point as part of support
        elif i == n - 1:
            supp_ids.append(i)
        else:
            if yc[i] != yc[i - 1]:
                supp_ids.append(i - 1)
                supp_ids.append(i)
    return np.unique(supp_ids)


def plot_layer12_feats(ds, args, params):
    """
    plot (c_j, ||b_j||) for model
    """
    if len(params) == 2:
        A, B = params
    else:
        A, B, _ = params

    kinks = -A[1, :] / A[0, :]
    # weights = np.abs(B) / np.abs(B).sum()

    weights = np.linalg.norm(B, axis=1)

    # plt.style.use('ggplot')
    plt.style.use('seaborn-colorblind')
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.scatter(kinks.flatten(), weights, marker='.', label=r"$(c_j, \lVert b_j \rVert)$")
    # plt.scatter(kinks.flatten(), weights, marker='.', color="#D55E00", label=r"$(c_j, \lVert b_j \rVert)$")
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    plt.xlim(-1, 1)
    plt.legend(loc='upper right', prop={'size': 22})
    if args['save']:
        plt.savefig(
            os.path.join(args['save_dir'],
                         "features.pdf")
        )
    else:
        plt.show()


