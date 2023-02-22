## Regression as Classification: Influence of Task Formulation on Neural Network Features

Proceedings of the 26th International Conference on Artificial
Intelligence and Statistics (AISTATS) 2023, Valencia, Spain.
PMLR: Volume 206. Copyright 2023 by the author(s)

[Link to paper on ArXiV](https://arxiv.org/abs/2211.05641)

#### Authors:
Lawrence Stewart (Ecole Normale Superieure and INRIA), Francis Bach (Ecole Normale Superieure and INRIA), Quentin Berthet (Google Brain), Jean-Philippe Vert (Google Brain).

#### Licence:
This code is distributed under a BSD-3 Licence.

#### Contents:

- `requirements.txt`: Lists all dependencies.
- `reparamfeats.py`: Creates Figure 2 from the paper.
- `optimalsupports.py`: Creates Figure 3 from the paper.
- `Rregmst.py`: Creates Figure 4 from the paper.
- `train.py` : Trains models on the synthetic 1D regression problem, detailed in Section 6.
-  `pooltasks.py`: Trains 30 random initializations and produces the results detailed in Section 6. 
-  `implbias2D.py`: Experiment detailed in Section 7.
-  `rmseplot.py`:  Generates Figure 5 (provided all runs are saved in `experiments` folder).
