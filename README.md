<img src="notebooks/assets/header_readme.gif" />
<hr>
<p align="center">
<b style="font-size:30vw;">Ocean subgrid parameterization in an indealized model using machine learning</b>
</p>
<hr>

This research aims to explore novel methods for parameterizing the contributions of subgrid-scale processes, which refer to physical phenomena occurring at scales finer than the simulation resolution. More precisely, this work is built upon the research of Ross et al., 2023, who, after many years of parameterization development, have created a framework to properly conduct the assessment of the quality of a parameterization.

In addition to replicating their findings, this study extends its scope by attempting to enhance their results through a series of experiments involving more complex datasets. Furthermore, and perhaps most significantly, it delves into the use of Fourier Neural Operators for modeling subgrid-scale process contributions. These neural networks were recently introduced by Li et al., 2020, and have already exhibited impressive results in many areas of computational fluid dynamics. Hence, while building upon the foundation laid by Ross et al., 2023, this study also pioneers the use of Fourier Neural Operators in this context, subjecting them to comprehensive evaluation within the established benchmarking framework.

In conclusion, this research not only facilitates a comprehensive grasp of the underlying physics in ocean-climate simulations but also delves into unexplored realms by leveraging state-of-the-art deep learning techniques for modeling subgrid-scale processes contributions. The conclusive results show promise and underscore the notion that the most captivating discoveries frequently emerge at the crossroads of two captivating scientific domains.

[1] Ross, Andrew et al. (2023). “Benchmarking of machine learning ocean subgrid parameterzations in an idealized model”. In: Journal of Advances in Modeling Earth Systems 15.1, e2022MS003258.

[2] Li, Zongyi et al. (2020). “Fourier neural operator for parametric partial differential equations”. In: arXiv preprint arXiv:2010.08895.

<hr>
<p  style="font-size:20px; font-weight:bold;" align="center">
Master thesis manuscript
</p>
<hr>
<img src="latex/TFE_Visual.pdf" />
<hr>
<p  style="font-size:20px; font-weight:bold;" align="center">
Results
</p>
<hr>

All the results are available [`here`](./results/results.pdf), they can also all be downloaded from the following <a href="https://drive.google.com/drive/folders/1hOfx6Ly6GrzDq_C_dq5hNgBTqMK8GAM8?usp=share_link" target="_blank" rel="noopener noreferrer">repository</a>.

<hr>
<p  style="font-size:20px; font-weight:bold;" align="center">
Installation
</p>
<hr>

1. Clone the repository

2. Create an appropriate **Conda** environnement:

```
conda env create -f environment.yml
```

3. Activate the  **Conda** environnement:

```
conda activate TFE
```

4. Install locally as a package:

```
pip install --editable .
```

5. Run the code easily using the notebook [`TFE.ipynb`](./notebooks/TFE.ipynb).