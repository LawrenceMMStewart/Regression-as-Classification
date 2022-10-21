import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 1000)

s1 = 1
k1 = -0.5

s2 = -1
k2 = 0.5

y1 = s1 * (x - k1)
y1[y1 < 0] = 0

y2 = s2 * (x - k2)
y2[y2 < 0] = 0


plt.style.use('seaborn-colorblind')
plt.figure()
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.plot(x, y1, label=r'$u_1=(1,-\frac{1}{2})$')
plt.plot(x, y2, label=r'$u_2=(-1,\frac{1}{2})$')
plt.legend(prop={'size' : 24})
plt.xlim(-1, 1 )
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel(r'$x$', fontsize=24)
plt.ylabel(r'$\phi_u(x)$', fontsize=24)
plt.show()
