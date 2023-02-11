#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#define exponential weighted cov matrix
def ewma_cov(returns, λ=0.97):
    N, T = returns.shape
    cov = np.zeros((N, N))
    weight = np.zeros((N, N))
    weight_sum = np.zeros((N, N))
    for t in range(T):
        weight = λ * weight + (1 - λ) * np.outer(returns[:, t], returns[:, t])
        weight_sum = weight_sum + weight
        cov = cov + weight
    cov = cov / (T - 1)
    weight_sum = weight_sum / T
    cov = cov / weight_sum
    return cov

def cumulative_variance_explained(pca):
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    return cum_var

# Load data
returns = pd.read_csv("DailyReturn.csv").values[:, 1:]


# Calculation of ewma covariance matrix for different λ
λs = [0.99, 0.97, 0.90, 0.8, 0.65, 0.4, 0.1]
for λ in λs:
    cov = ewma_cov(returns, λ)

    # Perform PCA on ewma covariance matrix
    pca = PCA().fit(cov)

    # Plot cumulative variance explained
    cum_var = cumulative_variance_explained(pca)
    x = np.arange(1, len(cum_var) + 1)
    plt.plot(x, cum_var, label=f"λ = {λ}")

plt.xlabel("Number of Eigenvalues")
plt.ylabel("Cumulative Variance Explained")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




