# GNN Acceleration

This project presents a comparative study of Graph Neural Network (GNN) acceleration techniques. We explore and evaluate two sampling methods and two sparsification methods:

### Sampling Methods
- **GraphSAGE**
- **GraphSAINT**

### Sparsification Methods
- **DropEdge**
- **NeuralSparse**

### Evaluation Parameters
The methods are compared based on the following parameters:
- **CPU Usage (%)**
- **GPU Usage (%)**
- **CPU Power Consumption (W)**
- **GPU Power Consumption (W)**
- **RAM Usage**
- **VRAM Usage**
- **Time per Epoch**
- **Number of Epochs**
- **ROC AUC**
- **EDP (Energy-Delay Product)**

This study aims to provide insights into the efficiency and performance trade-offs of different GNN acceleration techniques.

### Results
The results of the comparative study are given in the following table:

**TABLE I**: Machine Statistics while Training
| **Sparsification** <br> **Method** | **Sampling** <br> **Method** | **CPU** <br> **Usage (%)** | **CPU** <br> **Power (W)** | **GPU** <br> **Usage (%)** | **GPU** <br> **Power (W)** | **RAM** <br> **(MB)** | **vRAM** <br> **(MB)** |
|:----------------------------------:|:----------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:---------------------:|:----------------------:|
| None               | Uniformly Random | 29.6 | 10.2 | 4.3 | 5.5 | 4212 | 1450 |
| None               | GraphSAGE        | 40.9 | 18.6 | 15.6 | 5.0 | 6481 | 433 |
| None               | GraphSAINT       | 45.0 | 20.1 | 23.7 | 9.8 | 7121 | 1055 |
| DropEdge           | Uniformly Random | 40.3 | 18.0 | 42.8 | 20.0 | 3141 | 3388 |
| DropEdge           | GraphSAGE        | 44.0 | 19.5 | 41.8 | 19.3 | 6778 | 2560 |
| DropEdge           | GraphSAINT       | 47.2 | 21.2 | 23.9 | 9.3 | 7391 | 849 |
| NeuralSparse       | Uniformly Random | 36.5 | 14.2 | 27.4 | 16.4 | 3074 | 1639 |
| NeuralSparse       | GraphSAGE        | 38.9 | 16.0 | 48.9 | 21.6 | 8202 | 2955 |
| NeuralSparse       | GraphSAINT       | 36.8 | 14.6 | 22.9 | 9.1 | 7713 | 1943 |


**TABLE II**: Training Statistics
| **Sparsification** <br> **Method** | **Sampling** <br> **Method** | **Time per** <br> **Epoch** | **Number of** <br> **Epochs** | **ROC** <br> **AUC** | **EDP** |
| **Sparsification** <br> **Method** | **Sampling** <br> **Method** | **Time per** <br> **Epoch (s)** | **Number of** <br> **Epochs** | **ROC** <br> **AUC** | **EDP (W min²)** |
|:----------------------------------:|:----------------------------:|:---------------------------:|:----------------------------:|:--------------------:|:----------------:|
| None               | Uniformly Random | 18.1 | 49 | 66.56 | 3605.21 |
| None               | GraphSAGE        | 14.7 | 29 | 68.27 | 1726.45 |
| None               | GraphSAINT       | 1.4  | 45 | 72.23 | 48.29 |
| DropEdge           | Uniformly Random | 36.4 | 17 | 64.15 | 6913.80 |
| DropEdge           | GraphSAGE        | 28.2 | 38 | 74.76 | 19553.45 |
| DropEdge           | GraphSAINT       | 1.8  | 31 | 70.53 | **39.01** |
| NeuralSparse       | Uniformly Random | 106.2| 25 | 76.46 | 81455.40 |
| NeuralSparse       | GraphSAGE        | 84.1 | 19 | 77.34 | 46030.04 |
| NeuralSparse       | GraphSAINT       | 57.1 | 37 | **79.17** | 46494.81 |
