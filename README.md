# Unified GWO–ANN–SSN for Multiple Datasets

## Overview
This repository presents a **Unified GWO–ANN–SSN** framework that integrates a **Gray Wolf Optimizer (GWO)**, an **Artificial Neural Network (ANN)**, and a **Structured Sparsity Norm (SSN)** for efficient **feature selection** and **classification** across multiple image datasets.  
The approach automatically selects optimal features using GWO, evaluates subsets using an ANN with SSN regularization, and refines them iteratively to improve accuracy and reduce feature dimensionality.

---

## 📊 Flowchart of the Proposed Method

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                          START / DATASET SELECTION                       │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   DATASET DOWNLOAD & LOADING (KAGGLE API)                │
│        → Plants / AID / LC25000 datasets automatically fetched           │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    DEEP FEATURE EXTRACTION USING VGG16                   │
│       → Pretrained CNN used to extract high-level deep features          │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                DIMENSIONALITY REDUCTION VIA PCA (95% VARIANCE)           │
│       → Reduces redundant features while preserving essential info       │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            GRAY WOLF OPTIMIZER (GWO) – FEATURE OPTIMIZATION              │
│   → Initialize Wolf Population | α, β, δ Wolves | Hunting Mechanism      │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     ANN EVALUATION WITH SSN REGULARIZATION               │
│       → Computes Fitness using ANN + Structured Sparsity Norm (L₁₁, L₂₁) │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 OPTIMAL FEATURE SUBSET SELECTED BY GWO–ANN–SSN           │
│       → Alpha wolf position represents best feature subset               │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  FINAL ANN MODEL TRAINING & EVALUATION                   │
│       → Trains ANN on optimal features and evaluates test accuracy       │
└──────────────────────────────┬────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE METRICS & RESULT SAVING                   │
│     → Accuracy | Fitness | Selected Features | Reduction % | Summaries   │
└──────────────────────────────────────────────────────────────────────────┘
```
---

## 🧠 Key Concepts

- **Gray Wolf Optimizer (GWO):**  
  A metaheuristic optimization algorithm inspired by gray wolf hunting behavior, using alpha, beta, delta, and omega wolves for efficient feature subset selection.

- **Artificial Neural Network (ANN):**  
  A perceptron-based classifier used to evaluate the selected feature subsets.

- **Structured Sparsity Norm (SSN):**  
  Enforces sparsity and structure in network weights using L₁₁ and L₂₁ regularization norms for better feature discrimination.

---

## 📁 Supported Datasets
The framework supports three benchmark image datasets:

| Dataset   | Split Type | Description |
|------------|------------|--------------|
| **Plants** | Pre-split Train/Val/Test | Type classification of plant species |
| **AID** | 70–10–20 split | Aerial Image Dataset for scene classification |
| **LC25000** | Custom Split (80–20 + Test) | Lung and colon histopathological image dataset |

---

## ⚙️ Workflow

1. **Dataset Selection and Configuration**  
   Choose dataset (plants, aid, or lc25000) and environment (Colab or local).

2. **Automatic Download and Loading**  
   Downloads dataset from Kaggle using API credentials and loads into memory.

3. **Feature Extraction (VGG16)**  
   Pretrained VGG16 network extracts deep features for each image.

4. **Dimensionality Reduction (PCA)**  
   Reduces feature dimensionality while preserving 95% variance.

5. **GWO–ANN–SSN Feature Selection**  
   Uses GWO optimization, ANN classification, and SSN evaluation to identify optimal features.

6. **Final Model Training and Evaluation**  
   Trains final ANN model using selected features and evaluates on test set.

---

## 🧑‍💻 Code Execution

Follow the steps below to execute the **Unified GA–ANN–SSN** pipeline locally or in Colab.

### 🌐 1. Colab Execution

**Steps**
- Upload `Gwo_ann_ssn_3datasets.ipynb` file to **Google colab**
- Connect runtime to `T4 GPU`
- Run all cells 

**Advantages** of `Googel Colab` Execution
- Uses `T4 GPU`, No need of having GPU in local PC 
- Fast Execution
- No need of locally downloaded datasets

### 🖥️ 2. Local Execution

If you are running the project **locally**, make sure the following **dataset folders** are present in the **same directory** as your main script (`Ga_ann_ssn_3datasets.py`):

- 📁 GWO-ANN-SSN/
- │
- ├── Gwo_ann_ssn_3datasets.py
- ├── split_ttv_dataset_type_of_plants/
- ├── lung_colon_image_set/
- └── AID/


✅ **Required local dataset folders:**
- `split_ttv_dataset_type_of_plants/` → for **Plants dataset**
- `lung_colon_image_set/` → for **Lung & Colon dataset (LC25000)**
- `AID/` → for **Aerial Image Dataset (AID)**

Run the project using:
```bash
python Gwo_ann_ssn_3datasets.py
```

---
## 📈 Results & Performance

Below is the summary output from the **Unified GA–ANN–SSN** model execution across all supported datasets:

```text
======================================================================
GWO–ANN–SSN ALL DATASETS SUMMARY
======================================================================

Dataset      Test Acc   Val Acc    Features     Reduction    Classes 
----------------------------------------------------------------------
plants       0.9246     0.9743     171/331          48.3%         30      
aid          0.7600     1.0000     160/327          51.1%         30      
lc25000      0.9756     0.9916     148/325          54.5%         5 
======================================================================

```
---
## 💻 **Github Repo link:**
- https://github.com/codesbyYuvanraj/GA-ANN-SSN-ML-Model
