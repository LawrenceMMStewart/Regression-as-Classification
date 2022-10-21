import numpy as np
import matplotlib.pyplot as plt
from utils import get_bins, get_midpoints, regression_support, classification_support
from toydata import multiscale_triangle, get_bin_domains
from toydata import BalancedDataset


dseed = np.random.RandomState(52354)
cbins, obins = get_bins(50)
mids = get_midpoints(cbins)

ds = BalancedDataset(multiscale_triangle, 1 , 250, 250, cbins, dseed, reg=False)

reg_ids = regression_support(ds.x, ds.y)
class_ids = classification_support(ds.x, ds.yc)


print(f"R_reg has {len(reg_ids)} elements")
print(f"R_class has {len(class_ids)} elements")


plt.style.use('seaborn-colorblind')
plt.figure()
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.plot(ds.xt[:, 0], ds.yt, label=r'$f_{\mu_T}$', alpha=0.6)
plt.scatter(ds.x[:, 0], ds.y, marker='x', s=10, label='train data', zorder=3)
plt.scatter(ds.x[reg_ids, 0], ds.y[reg_ids, 0], marker=',', alpha=0.5, color='orange', label=r"$R_{reg}$", s=60)
plt.ylim(0 - 0.1, 1 + 0.1)
plt.xlim(-1 - 0.1, 1 + 0.1)
plt.legend(loc='lower left', prop={'size': 20})
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

