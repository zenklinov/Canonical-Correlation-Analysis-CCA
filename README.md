# Canonical Correlation Analysis (CCA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains implementations and business use cases of **Canonical Correlation Analysis (CCA)** using Python. CCA is a multivariate statistical method used to identify and measure the associations between two sets of variables.

🔗 **Repository:** [https://github.com/zenklinov/Canonical-Correlation-Analysis-CCA](https://github.com/zenklinov/Canonical-Correlation-Analysis-CCA)

## Table of Contents
- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
- [Repository Structure](#repository-structure)
- [Installation & Prerequisites](#installation--prerequisites)
- [Case Studies](#case-studies)
    - [1. Linnerud Dataset (Physiological vs. Exercise)](#1-linnerud-dataset-physiological-vs-exercise)
    - [2. Business Case (Marketing vs. Sales)](#2-business-case-marketing-vs-sales)
- [Implementation Code](#implementation-code)

## Overview
While standard correlation analyzes the relationship between two single variables ($X$ and $Y$), **CCA** analyzes the relationship between two **sets** of variables ($X_1, ... X_n$ and $Y_1, ... Y_m$).

It determines a set of linear combinations (called **Canonical Variates**) for each set of variables such that the correlation between these variates is maximized.

## Mathematical Formulation

Given two column vectors $X = (x_1, \dots, x_n)^T$ and $Y = (y_1, \dots, y_m)^T$ of random variables.

We seek vectors $a$ (weights for X) and $b$ (weights for Y) such that the random variables $U$ and $V$:

$$U = a^T X$$
$$V = b^T Y$$

maximize the correlation $\rho$:

$$\rho = \text{corr}(U, V) = \frac{\text{cov}(U, V)}{\sqrt{\text{var}(U) \cdot \text{var}(V)}}$$

Where:
* $U$ and $V$ are the **Canonical Variates**.
* $a$ and $b$ are the **Canonical Weights** (or coefficients).
* $\rho$ is the **Canonical Correlation**.

## Repository Structure

| File Name | Description |
| :--- | :--- |
| `Canonical-Correlation-Analysis-CCA.ipynb` | A fundamental introduction to CCA using the standard Linnerud dataset (Exercise vs. Physiological metrics). |
| `CCA-Business-Case-Marketing-vs-Sales.ipynb` | A synthetic business simulation analyzing how different Marketing Investments impact specific Sales Outcomes. |

## Installation & Prerequisites

To run these notebooks, you need Python installed along with the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
## Case Studies

### 1. Linnerud Dataset (Physiological vs. Exercise)
*File: `Canonical-Correlation-Analysis-CCA.ipynb`*

We analyze the relationship between three exercise variables and three physiological measurements from 20 middle-aged men.

* **Set X (Exercise):** Chins, Situps, Jumps
* **Set Y (Physiological):** Weight, Waist, Pulse

**Key Findings:**
* **Strong Correlation (0.80):** There is a strong link between exercise performance and body metrics.
* **Insight:** Men who performed more **Situps** tended to have smaller **Waist** measurements.

### 2. Business Case (Marketing vs. Sales)
*File: `CCA-Business-Case-Marketing-vs-Sales.ipynb`*

A simulated business scenario to answer: *"How do different marketing channels drive specific business outcomes?"*

* **Set X (Investments):** TV Spend, Social Media Spend, Email Spend.
* **Set Y (Outcomes):** In-Store Sales, Online Sales, Retention Rate.

**Key Findings:**
* **Strategy 1 (Digital):** High `Social` & `Email` spend correlates strongly with `Online Sales` and `Retention`.
* **Strategy 2 (Traditional):** High `TV Spend` correlates strongly with `In-Store Sales`.
* CCA allows the business to disentangle these mixed effects to optimize budget allocation.

## Implementation Code

Below is a minimal example of how to implement CCA using `scikit-learn`:

```python
from sklearn.cross_decomposition import CCA
import numpy as np

# 1. Prepare Data
# X: 2 variables, Y: 2 variables
X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]

# 2. Initialize CCA object
# n_components is the number of canonical pairs to find
cca = CCA(n_components=1)

# 3. Fit the model
cca.fit(X, Y)

# 4. Transform data to get Canonical Variates
X_c, Y_c = cca.transform(X, Y)

# 5. Check Correlation
result = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
print(f"Canonical Correlation: {result}")
```
