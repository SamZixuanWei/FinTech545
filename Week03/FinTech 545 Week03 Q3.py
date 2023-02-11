#!/usr/bin/env python
# coding: utf-8

# In[350]:


import numpy as np
from scipy.linalg import eigh
import pandas as pd
from scipy.signal import resample
import csv

df = pd.read_csv("DailyReturn.csv")

# Replace zeros with a small positive value
df = df.replace(0, 1e-5)

# Calculate mean and covariance of the data
mean = df.mean().values
covariance = df.cov().values
def multivariate_normal_simulation(mean, covariance, pca=False, explained_variance=None):
    n = mean.shape[0]
    covariance[np.diag_indices(n)] += 1e-8 # Adding a small positive value to the diagonal to handle zero values
    if pca:
        eigenvalues, eigenvectors = eigh(covariance)
        explained_variance = explained_variance if explained_variance is not None else 0.95
        explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        num_components = np.searchsorted(explained_variance, explained_variance) + 1
        eigenvectors = eigenvectors[:, -num_components:]
        transformed_data = np.random.normal(size=(num_components, n))
        data = np.dot(eigenvectors, transformed_data.T).T + mean
    else:
        L = np.linalg.cholesky(covariance)
        data = np.dot(L, np.random.normal(size=mean.shape)) + mean
    return data



simulated_data = multivariate_normal_simulation(mean, covariance, pca=False)

def generate_correlation_matrix(df):
        correlation_matrix = df.corr()
        return correlation_matrix
    
def generate_correlation_matrix_exp(returns):
    n_stocks = returns.shape[1]
    corr_matrix = np.zeros((n_stocks, n_stocks))
    weight = np.zeros(returns.shape[0])
    weight[0] = 1
    for i in range(1, returns.shape[0]):
        weight[i] = weight[i-1] * lambda_value

    for i in range(n_stocks):
        for j in range(i, n_stocks):
            weighted_cov = np.sum(weight * (returns[:,i] - returns.mean(axis=0)[i]) * (returns[:,j] - returns.mean(axis=0)[j])) / np.sum(weight)
            corr_matrix[i][j] = weighted_cov / (returns.std(axis=0)[i] * returns.std(axis=0)[j])
            corr_matrix[j][i] = corr_matrix[i][j]
    return corr_matrix



def generate_variance_vector_exp(filename, lambda_value):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip the header row
        weighted_average = [0.0] * len(next(reader)) # initialize weighted average and variance
        weighted_variance = [0.0] * len(weighted_average)
        for i, row in enumerate(reader):
            for j, data in enumerate(row[1:]):
                data = float(data)
                weighted_average[j] = lambda_value * weighted_average[j] + (1 - lambda_value) * data
                weighted_variance[j] = lambda_value * weighted_variance[j] + (1 - lambda_value) * (data - weighted_average[j])**2
    return weighted_variance
 

def covariance_matrix(correlation_matrice,variance_vectors):
    covariance_matrice = correlation_matrice * variance_vectors
    return covariance_matrice


correlation_matrix_pearson = generate_correlation_matrix(df)
variance_vector_pearson = generate_variance_vector(df)
returns_data = pd.read_csv("DailyReturn.csv", index_col=0)
returns = returns_data.values
correlation_matrix_exponential = generate_correlation_matrix_exp(returns)  
stock_names = ["SPY","AAPL","MSFT","AMZN","TSLA","GOOGL","GOOG","FB","NVDA","BRK-B","JPM","JNJ","UNH","HD","PG","V","BAC","MA","PFE","XOM","DIS","CSCO","AVGO","ADBE","CVX","PEP","TMO","KO","ABBV","CMCSA","NFLX","ABT","ACN","COST","CRM","INTC","WFC","VZ","PYPL","WMT","QCOM","MRK","LLY","MCD","T","NKE","DHR","LOW","LIN","TXN","NEE","AMD","UNP","PM","INTU","UPS","HON","MS","MDT","BMY","AMAT","ORCL","SCHW","CVS","RTX","C","GS","AMGN","BLK","BA","CAT","IBM","SBUX","AMT","PLD","GE","ISRG","COP","TGT","ANTM","AXP","DE","MU","SPGI","MMM","NOW","BKNG","F","ADP","ZTS","LRCX","PNC","MDLZ","MO","ADI","GILD","LMT","SYK","GM","TFC","TJX"]   
correlation_matrix_exponential = pd.DataFrame(correlation_matrix_exponential,columns=stock_names,index=stock_names)
filename = "DailyReturn.csv"
lambda_value = 0.97
result = generate_variance_vector_exp(filename, lam)
item = result.pop()
variance_vector_exponential = list(zip(stock_names,result))
variance_vector_exponential = pd.Series(dict(variance_vector_exponential))

covariance_matrix_pearson_pearson = covariance_matrix(correlation_matrix_pearson,variance_vector_pearson)
covariance_matrix_exponential_pearson = covariance_matrix(correlation_matrix_exponential,variance_vector_pearson)
covariance_matrix_exponential_exponential = covariance_matrix(correlation_matrix_exponential,variance_vector_exponential)
covariance_matrix_pearson_exponential = covariance_matrix(correlation_matrix_pearson,variance_vector_exponential)

print(covariance_matrix_exponential_pearson)


# In[360]:


def direct_simulation(cov_matrix, num_samples):
    mean = np.zeros(cov_matrix.shape[0]) # mean is set to zero for this example
    return np.random.multivariate_normal(mean, cov_matrix, num_samples)

cov_matrix = covariance_matrix_pearson_pearson
num_samples = 25000

samples_p_p = direct_simulation(cov_matrix, num_samples)
cov_matrix = covariance_matrix_pearson_exponential
samples_p_e = direct_simulation(cov_matrix, num_samples)
cov_matrix = covariance_matrix_exponential_pearson
samples_e_p = direct_simulation(cov_matrix, num_samples)
cov_matrix = covariance_matrix_exponential_exponential
samples_e_e = direct_simulation(cov_matrix, num_samples)


# In[367]:


from sklearn.decomposition import PCA

def pca_simulation(cov_matrix, num_samples, explained_variance):
    mean = np.zeros(cov_matrix.shape[0]) # mean is set to zero for this example
    pca = PCA(n_components=cov_matrix.shape[0], svd_solver='full')
    pca.fit(cov_matrix)
    components = pca.components_
    explained_variance = pca.explained_variance_
    return np.random.multivariate_normal(mean, cov_matrix, num_samples).dot(components.T) * np.sqrt(explained_variance)

cov_matrix = covariance_matrix_pearson_pearson
num_samples = 25000
explained_variance = 1.0

samples_p_p_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_pearson_exponential
samples_p_e_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_exponential_pearson
samples_e_p_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_exponential_exponential
samples_e_e_1 = pca_simulation(cov_matrix, num_samples, explained_variance)


explained_variance = 0.75
cov_matrix = covariance_matrix_pearson_pearson
samples_p_p_075 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_pearson_exponential
samples_p_e_075 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_exponential_pearson
samples_e_p_075 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_exponential_exponential
samples_e_e_075 = pca_simulation(cov_matrix, num_samples, explained_variance)


explained_variance = 0.5
cov_matrix = covariance_matrix_pearson_pearson
samples_p_p_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_pearson_exponential
samples_p_e_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_exponential_pearson
samples_e_p_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
cov_matrix = covariance_matrix_exponential_exponential
samples_e_e_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
print(samples_e_p_05)


# In[329]:


input_cov = covariance_matrix_pearson_pearson.values
simulated_cov = samples_p_p
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
print("Frobenius Norm: ", frobenius_norm)
input_cov = covariance_matrix_pearson_exponential.values
simulated_cov = samples_p_e
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
print("Frobenius Norm: ", frobenius_norm)
input_cov = covariance_matrix_exponential_pearson.values
simulated_cov = samples_e_p
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_exponential_exponential.values
simulated_cov = samples_e_e
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_pearson_pearson.values
simulated_cov = samples_p_p_1
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_pearson_exponential.values
simulated_cov = samples_p_e_1
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_exponential_pearson.values
simulated_cov = samples_e_p_1
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_exponential_exponential.values
simulated_cov = samples_e_e_1
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_pearson_pearson.values
simulated_cov = samples_p_p_075
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_pearson_exponential.values
simulated_cov = samples_p_e_075
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_exponential_pearson.values
simulated_cov = samples_e_p_075
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_exponential_exponential.values
simulated_cov = samples_e_e_075
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_pearson_pearson.values
simulated_cov = samples_p_p_05
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_pearson_exponential.values
simulated_cov = samples_p_e_05
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_exponential_pearson.values
simulated_cov = samples_e_p_05
print("Frobenius Norm: ", frobenius_norm)
frobenius_norm = np.linalg.norm(input_cov - simulated_cov, ord='fro')
input_cov = covariance_matrix_exponential_exponential.values
simulated_cov = samples_e_e_05
print("Frobenius Norm: ", frobenius_norm)


# In[373]:


import time
start_time = time.time()

cov_matrix = covariance_matrix_pearson_pearson
num_samples = 25000
explained_variance = 1.0
samples_p_p_1 = pca_simulation(cov_matrix, num_samples, explained_variance)

run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[374]:


start_time = time.time()

cov_matrix = covariance_matrix_pearson_pearson
num_samples = 25000
samples_p_p = direct_simulation(cov_matrix, num_samples)

run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[375]:


start_time = time.time()

cov_matrix = covariance_matrix_pearson_exponential
num_samples = 25000
samples_p_e = direct_simulation(cov_matrix, num_samples)

run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[376]:


start_time = time.time()

cov_matrix = covariance_matrix_exponential_pearson
num_samples = 25000
samples_e_p = direct_simulation(cov_matrix, num_samples)

run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[377]:


start_time = time.time()

cov_matrix = covariance_matrix_exponential_exponential
num_samples = 25000
samples_e_e = direct_simulation(cov_matrix, num_samples)

run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[397]:


start_time = time.time()
cov_matrix = covariance_matrix_pearson_pearson
num_samples = 25000
explained_variance = 1.0
samples_p_p_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[379]:


start_time = time.time()
cov_matrix = covariance_matrix_pearson_exponential
num_samples = 25000
explained_variance = 1.0
samples_p_e_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[399]:


start_time = time.time()
cov_matrix = covariance_matrix_exponential_exponential
num_samples = 25000
explained_variance = 1.0
samples_e_e_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[381]:


start_time = time.time()
cov_matrix = covariance_matrix_exponential_pearson
num_samples = 25000
explained_variance = 1.0
samples_e_p_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[382]:


start_time = time.time()
cov_matrix = covariance_matrix_exponential_exponential
num_samples = 25000
explained_variance = 1.0
samples_e_e_1 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[383]:


start_time = time.time()
cov_matrix = covariance_matrix_pearson_pearson
num_samples = 25000
explained_variance = 0.75
samples_p_p_075 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[402]:


start_time = time.time()
cov_matrix = covariance_matrix_pearson_exponential
num_samples = 25000
explained_variance = 0.75
samples_p_e_075 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[385]:


start_time = time.time()
cov_matrix = covariance_matrix_exponential_pearson
num_samples = 25000
explained_variance = 0.75
samples_e_p_075 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[386]:


start_time = time.time()
cov_matrix = covariance_matrix_exponential_exponential
num_samples = 25000
explained_variance = 0.75
samples_e_e_075 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[387]:


start_time = time.time()
cov_matrix = covariance_matrix_pearson_pearson
num_samples = 25000
explained_variance = 0.5
samples_p_p_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[388]:


start_time = time.time()
cov_matrix = covariance_matrix_pearson_exponential
num_samples = 25000
explained_variance = 0.5
samples_p_e_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[389]:


start_time = time.time()
cov_matrix = covariance_matrix_exponential_pearson
num_samples = 25000
explained_variance = 0.5
samples_e_p_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[390]:


start_time = time.time()
cov_matrix = covariance_matrix_exponential_exponential
num_samples = 25000
explained_variance = 0.5
samples_e_e_05 = pca_simulation(cov_matrix, num_samples, explained_variance)
run_time = time.time() - start_time
print("Run time:", run_time, "seconds")


# In[ ]:





# In[ ]:




