# Project: Data-Quality-Filter-A-New-Approach-For-Edge-Intelligence

# Article: Federated Learning Performance Evaluation with Entropy-Based Data Quality Analysis Adoption

Algorithms used in training the article's datasets, obtaining information for the Enbase approach.

# EnBaSe (Entropy-Based Selection): Algorithm Using Entropy for Data Filtering

**EnBaSe** is the algorithm that served as the basis for this research, incorporating the use of **entropy** in its training.  

This algorithm stands out for its ability to **filter model data** by calculating entropy, a measure that quantifies the uncertainty associated with a set of information.  

## How it works
- EnBaSe calculates the entropy of the data and establishes an **average uncertainty**.  
- This allows it to **identify which sets of information are most reliable and relevant** for analysis.  

## Benefits
- **Selection of data with less uncertainty**.  
- **Reduced complexity**: less information is needed to generate effective results.  
- **Simplification of the analysis and interpretation process**.

## ⚙️ Configurations Used
During the experiments conducted in the research, the following configurations were used:   
- **Language**: Python 3.x
- **Main libraries**: NumPy, PyTorch/TensorFlow
- **Environment**: Jupyter Notebook / Google Colab
- **Hardware**: NVIDIA GPU (CUDA enabled) for training optimization - using Tesla4 and A100 GPU.
- **Main hyperparameters**:
  - Learning rate: *0.001*
  - Batch size: *64*
  - Number of epochs: *50*
  - Optimizer: *Adam*  

---

## 🧮 Algorithms Used in Training
**EnBaSe** was based on some recurrent neural networks such as:

- **Convolutional Neural Networks (CNNs)**
Mainly applied in image classification tasks.  

- **Recurrent Neural Networks (RNNs/LSTMs)**  
  Used for sequential data and temporal analysis.

---

## 🏋️‍♂️ Training Methods: Normal Training vs. EnBaSe Training
In this research, two distinct model training methods were employed:  

### 🔹 Normal Training
- Conventional training process.
- Uses all data from the dataset without additional filtering.
- Maintains **complete data variability**, but may include noise and samples with high uncertainty.  

### 🔹 EnBaSe Training
- Training based on **entropy** filtering.  
- The **EnBaSe** algorithm selects only data with the **lowest level of uncertainty**.  
- Results in more **optimized, consistent, and robust** models in the face of noise.  

---

## 📊 Model Creation
A total of **12 models** were created, distributed between the two training methods, using different datasets:

- **Normal Training (6 models)**
  - MNIST
  - CIFAR-10  
  - CIFAR-100  
  - Fashion-MNIST  
  - Caltech101  
  - EuroSAT  

- **EnBaSe Training (6 models)**  
  - MNIST  
  - CIFAR-10  
  - CIFAR-100  
  - Fashion-MNIST  
  - Caltech101  
  - EuroSAT

Each dataset was trained in **two versions** (Normal and EnBaSe), enabling direct comparison of the results.

---

## Results Achieved

- Accuracy Average Results, using both T4 and A100 GPUs, in the two proposed scenarios: Normal Training and EnBaSe Training.

| **GPU**                | **MNIST** | **CIFAR-10** | **CIFAR-100** | **FASHION-MNIST** | **Caltech101** | **EuroSAT** |
|-------------------------|-----------|---------------|----------------|-------------------|----------------|--------------|
| T4 (Normal Training)    | 0.8172    | 0.9690        | 0.9264         | 0.7439            | 0.9831         | 0.9876       |
| A100 (Normal Training)  | 0.8609    | 0.9728        | 0.9296         | 0.9482            | 0.9940         | 0.9948       |
| T4 (Enbase Training)    | 0.9997    | 0.9346        | 0.9350         | 0.9942            | 0.7736         | 0.9543       |
| A100 (Enbase Training)  | 0.9997    | 0.9399        | 0.9472         | 0.9941            | 0.6100         | 0.9474       |

**Tabela 1.** Average Accuracy Comparison

📌 This repository aims to present the concepts, functioning, and potential applications of **EnBaSe** in the context of data filtering and optimization.
