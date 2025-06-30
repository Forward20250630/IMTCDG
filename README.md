# Integrated Multidomain Temporal Contrastive Domain Generalization

Official PyTorch implementation for "A novel integrated multidomain temporal contrastive domain generalization learning framework for time-series quality prediction of batch machining" 

## Introduction
Time-series quality prediction has been a challenging problem due to insufficient corresponding data and inconsistent temporal data distribution. Traditional time-series quality prediction methods requires that future data distribution is consistent with historical data distribution, which limits its practical applications in batch machining systems. Hence, a novel integrated multidomain temporal contrastive domain generalization learning framework of batch machining systems is proposed to capture temporal invariant relationship to achieve accurate time-series quality prediction with temporal data distribution shift. Specifically, the auxiliary domain is integrated to improve data diversity by matching source domain with other out-of-distribution multidomain based on temporal data distribution similarity and machining time segment division. The multi-level temporal contrastive domain generalization effectively obtains temporal invariant relationships between historical and future machining periods to accurately predict future unknown target domain. Global-Local temporal contrastive learning is combined with dynamic intra- and inter-domain Adversarial Spectral Kernel Matching to enhance the capability for temporal invariant features extraction. Experiment results on a batch CNC machining system show that compared to other domain generalization methods, the proposed method has an average improvement of 8.19%, 8.15%, and 10.44% in MAE, RMSE, and R2, respectively.

## Environment
- python 3.9.12
- numpy 1.21.5
- pandas 1.4.2
- matplotlib 3.5.1
- scikit-learn 1.1.2
- torch 1.12.0+cu116

## Datasets
The model running relies on the time-series machining dataset of the main source domain, auxiliary domain, and unknown target domain. The dataset employed in this paper cannot be made publicly available due to the institute's non-disclosure agreement and copyright restrictions.

## Usage
1. Set up a Python environment according to requirements.
2. Load datasets and configure file paths.
3. Construct the auiliary domain by applying methods in folder `./AuxiliarydomainConstruction/`
4. In `main.py`, configure file paths separately for: main source domain, auxiliary domain, and unknown target domain datasets.
5. Set model hyperparameters and run main.py


<!--
**IMTCDG/IMTCDG** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
