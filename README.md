# Interbank

Interbank Risk and Credit Rating: Datasets and Methods

## Overview

Source codes implementation of papers:

- Credit Rating Method:
  
## Contents

- [Repo Structure](#repo-structure)
- [Dataset](#dataset)
- [Methods](#Methods)
  - [credit_rating](#credit_rating)
      -[baseline](#baseline)
      -[graph_model](#graph_model)
      -[utils](#utils)

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
python                       3.7.16
torch                        1.13.1
torch-cluster                1.6.1
pyg                          2.3.0
torch-scatter                2.1.1
torch-sparse                 0.6.17
tqdm                         4.42.1
scikit-learn                 1.0.2
pandas                       1.2.3
numpy                        1.21.5
```


## 
