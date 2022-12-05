## Regression as Classification: Influence of Task Formulation on Neural Network Features

Code to reproduce the paper Regression as Classification: Influence of Task Formulation on Neural Network Features:  Lawrence Stewart (Ecole Normale Superieure), Francis Bach (Ecole Normale Superieure), Quentin Berthet (Google Brain), Jean-Philippe Vert (Google Brain).

https://arxiv.org/abs/2211.05641


- `reparamfeats.py`: Creates Figure 2 from the paper.
- `optimalsupports.py`: Creates Figure 3 from the paper.
- `Rregmst.py`: Creates Figure 4 from the paper.
- `train.py` : Trains models on the synthetic 1D regression problem, detailed in Section 6.
-  `pooltasks.py`: Trains 30 random initializations and produces the results detailed in Section 6. 
-  `implbias2D.py`: Experiment detailed in Section 7.
-  `rmseplot.py`:  Generates Figure 5 (provided all runs are saved in `experiments` folder).
