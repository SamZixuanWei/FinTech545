#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import scipy
import copy
from bisect import bisect_left


# Covariance estimation techniques

# exponentially weighted covariance matrix
def exp_weighted_cov(returns, lambda_=0.97): 
    returns = returns.to_numpy()
    mean_return = np.mean(returns, axis=0)
    normalized_returns = returns - mean_return
    n_timesteps = normalized_returns.shape[0]
    cov = np.cov(normalized_returns, rowvar=False)
    for i in range(1, n_timesteps):
        cov = lambda_ * cov + (1 - lambda_) * np.outer(normalized_returns[i], normalized_returns[i])
    
    return cov


# Non PSD fixes for correlation matrices 
def near_psd(a, epsilon=0.0):
    n = a.shape[1]
    out = a.copy()
    
    # Ensure the matrix is symmetric
    out = (out + out.T) / 2.0
    
    # Ensure the matrix is positive semi-definite
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    l = np.diag(np.sqrt(vals))
    B = vecs @ l @ vecs.T
    out = (B + B.T) / 2.0
    
    return out


def higham_psd(a, max_iter=100, tol=1e-8):
    delta_s = 0.0
    y = np.copy(a)
    prev_gamma = np.inf
    
    for i in range(max_iter):
        r = y - delta_s
        x = projection_s(r)
        delta_s = x - r
        y = projection_u(x)
        gamma = np.linalg.norm(y - a, ord='fro')
        
        if abs(gamma - prev_gamma) < tol:
            break
        
        prev_gamma = gamma
    
    return y

def projection_s(a):
    w, v = np.linalg.eigh(a)
    w = np.maximum(w, 0.0)
    return v @ np.diag(w) @ v.T

def projection_u(a):
    return (a + a.T) / 2.0

# Simulation Methods
def chol_psd(a):
    n = a.shape[1]
    root = np.zeros_like(a)

    for j in range(n):
        s = root[:j, :j] @ root[:j, j]
        temp = a[j, j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0:
            root[j, j:n] = 0.0
        else:
            root[j+1:n, j] = (a[j+1:n, j] - root[j+1:n, :j] @ root[j, :j].T) / root[j, j]

    return root[:n, :n]

def direct_simulation(cov, n_samples=100):
    # Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov)
    # Generate n_samples independent standard normal random vectors
    Z = np.random.randn(cov.shape[0], n_samples)
    # Transform the samples to the correlated multivariate normal distribution
    X = L @ Z
    return X

def pca_simulation(cov, pct_explained, n_samples=100):
    eigen_values, eigen_vectors = np.linalg.eigh(cov)
    # sort eigenvalues and eigenvectors in descending order
    sorted_idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[sorted_idx]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    # calculate explained variance ratio
    explained_var = eigen_values / np.sum(eigen_values)
    # calculate cumulative explained variance ratio
    cum_explained_var = np.cumsum(explained_var)
    # clip values greater than 1 to 1
    cum_explained_var = np.clip(cum_explained_var, 0, 1)
    # find the index for the given percentage of explained variance
    idx = np.argmax(cum_explained_var >= pct_explained)
    # select the eigenvectors and eigenvalues up to the index
    selected_vectors = eigen_vectors[:, :idx+1]
    selected_values = eigen_values[:idx+1]
    # compute B matrix and generate random samples
    B = selected_vectors @ np.diag(np.sqrt(selected_values))
    r = np.random.randn(B.shape[1], n_samples)
    return B @ r

# VaR calculation methods (all discussed)
def calculate_var(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)

def normal_var(data, mean=0, alpha=0.05, nsamples=100):
    sigma = np.std(data)
    simulation_norm = np.random.normal(mean, sigma, nsamples)
    var_norm = calculate_var(simulation_norm, mean, alpha)
    return var_norm

def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    ew_cov = calculate_ewcov(np.matrix(data).T, 0.94)
    ew_variance = ew_cov[0, 0]
    sigma = np.sqrt(ew_variance)
    simulation_ew = np.random.normal(mean, sigma, nsamples)
    var_ew = calculate_var(simulation_ew, mean, alpha)
    return var_ew

def t_var(data, mean=0, alpha=0.05, nsamples=100):
    df, loc, scale = scipy.stats.t.fit(data, method="MLE")
    simulation_t = scipy.stats.t(df, loc, scale).rvs((nsamples,))
    var_t = calculate_var(simulation_t, mean, alpha)
    return var_t

def ar1_var(mu, sigma, rho, alpha, alpha_level, timesteps=1000, n_simulations=100):
    # Calculate the standard deviation of the AR(1) process
    ar1_std = np.sqrt(sigma**2 / (1 - alpha**2))

    # Simulate n_simulations paths of the AR(1) process
    ar1_paths = np.zeros((n_simulations, timesteps))
    ar1_paths[:,0] = norm.ppf(np.random.rand(n_simulations)) * ar1_std
    for t in range(1, timesteps):
        ar1_paths[:,t] = mu + alpha * ar1_paths[:,t-1] + norm.ppf(np.random.rand(n_simulations)) * sigma

    # Calculate the VaR at the specified alpha_level
    ar1_quantiles = np.quantile(ar1_paths[-1,:], alpha_level)
    return ar1_quantiles

def historic_var(data, mean=0, alpha=0.05):
    return calculate_var(data, mean, alpha)

# ES calculation

def calculate_es(data, mean=0, alpha=0.05):
    var = calculate_var(data, mean, alpha)
    es = -np.mean(data[data <= -var])
    return es


# In[ ]:





# In[ ]:




