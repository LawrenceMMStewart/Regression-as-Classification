import numpy as np
import matplotlib.pyplot as plt
from utils import get_bins, get_midpoints, regression_support, classification_support
from toydata import multiscale_triangle, get_bin_domains
from toydata import BalancedDataset


cbins, obins = get_bins(5)
mids = get_midpoints(cbins)

fn = lambda x: np.abs(x)

x = np.linspace(-1, 1, 30).reshape(-1, 1)
xt = np.linspace(-1, 1, 10000).reshape(-1, 1)
y = fn(x)
yt = fn(xt)
x = np.concatenate((x, np.ones_like(x)), axis=1)
xt = np.concatenate((xt, np.ones_like(xt)), axis=1)
reg_ids = regression_support(x, y)
class_ids = classification_support(x, np.digitize(y, cbins))

plt.style.use('seaborn-colorblind')
plt.figure()
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.plot(xt[:, 0], yt, label=r'$f(x) = \lvert x \rvert$', alpha=0.4)
plt.scatter(x[:, 0], y,  marker='x', s=10, label=r'Data $\{(x_i,y_i)\}_{i=1}^n$', alpha=0.9, zorder=3)
plt.scatter(x[class_ids, 0], y[class_ids, 0], marker='v', alpha=0.7, label="$R_{class}$", s=40, zorder=2)
plt.scatter(x[reg_ids, 0], y[reg_ids, 0], marker=',', alpha=0.5, label=r"$R_{reg}$", s=60)
for b in cbins[:-1]:
     plt.axhline(y=b, linestyle='-', alpha=0.4, color='r')
plt.axhline(y=cbins[-1], linestyle='-', alpha=0.4, label="bins", color='r')
plt.legend(prop={'size' : 14})
plt.ylim(0 - 0.1, 1 + 0.1)
plt.xlim(-1 - 0.1, 1 + 0.1)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.show()
