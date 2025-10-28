# Interbank

Interbank Risk and Credit Rating: Datasets and Methods

## Overview

This repository contains the codes of the master thesis: A GNN-Based Framework for Bank Creditworthiness Prediction Using Interbank Networks by Francesco Salvagnin
  
## Contents

- [Repo Structure](#repo-structure)
- [Dataset](#dataset)
- [Train](#Train)
- [Requirements](#Requirements)
  
## Repo Structure

`datasets/:` dataset files;

`methods/:` the implementation of interbank credit rating methods

## Dataset

The Dataset is collected from the GitHub repository: https://github.com/AI4Risk/interbank
## Methods

Contains all the methods used for train the baseline models and the TGAR model

### Train

To train the **TGAR** model, run

```
python -m methods.credit_rating.graph_model.train --net [TGAR] --year [predict_year] --quarter [predict_quarter] --epochs [epoch] 
```

To train the **LSTM** model, run
```
!python -m methods.credit_rating.baseline.run_lstm --start_quarter [first_year] --end_quarter[last_year] --EPOCHS[epochs] 
```

To train the **MLP** model, run
```
!python -m methods.credit_rating.baseline.run_mlp --start_quarter [first_year] --end_quarter[last_year] --EPOCHS[epochs]
```


## Requirements

```
python                       3.12.12
torch-geometric              2.5.3
torch                        2.5.1  
pyg_lib                      0.4.0
torch-scatter                2.1.1
torch-sparse                 1.6.3
scikit-learn                 1.6.1
pandas                       2.2.2
numpy                        2.3.4
```
