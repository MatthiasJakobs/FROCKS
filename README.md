<div align="center">

# Federated Time Series Classification with ROCKET features
Bruno Casella, Matthias Jakobs, Marco Aldinucci, Sebastian Buschjäger

[![Conference](https://img.shields.io/badge/ESANN-2024-orange)]([add_link](https://doi.org/10.14428/esann/2024.ES2024-61))

</div>

# Overview

This repository contains the code to run and reproduce the experiments for federating the ROCKET algorithm in the following settings:
- Federated baseline
- FROCKS

Please cite as:

```bibtex
@inproceedings{casella2024frocks,
  author  = {Casella, Bruno and Jakobs, Matthias and Aldinucci, Marco and Buschjager, Sebastian},
  title   = {Federated Time Series Classification with ROCKET features,
  booktitle    = {32nd European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, {ESANN} 2024, Bruges, Belgium, October 9-11, 2024},
  year         = {2024},
  doi          = {10.14428/ESANN/2024.ES2024-61},
  url = {https://doi.org/10.14428/esann/2024.ES2024-61}
}
```

# Abstract
This paper proposes FROCKS, a federated time series classification method using ROCKET features. Our approach dynamically adapts the models’ features by selecting and exchanging the best-performing ROCKET kernels from a federation of clients. Specifically, the server gathers the best-performing kernels of the clients together with the associated model parameters, and it performs a weighted average if a kernel is best-performing for more than one client. We compare the proposed method with state-of-the-art approaches on the UCR archive binary classification datasets and show superior performance on most datasets.

## Usage
- Clone this repo.
- Download and unzip the UCRArchive Time Series Classification dataset: [UCRArchive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), and place it in the same directory of the repository.
- Install the requirements. 
- Each folder of this repository contains specific instructions on how to execute the centralized, federated, and FROCKS experiments.

## Results
The results directory reports the accuracy and F1-score metrics for all the UCR Archive binary classification datasets. This folder contains also the critical difference diagrams, a useful tool for comparing the performance of the naive and FROCKS methods over all the considered datasets.


## Contributors
* Bruno Casella <bruno.casella@unito.it>  

* Matthias Jakobs <matthias.jakobs@tu-dortmund.de>

* Marco Aldinucci <marco.aldinucci@unito.it>

* Sebastian Buschjäger <sebastian.buschjaeger@tu-dortmund.de>
