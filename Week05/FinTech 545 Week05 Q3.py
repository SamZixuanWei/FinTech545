#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.stats import kurtosis
from scipy.optimize import minimize
import matplotlib.pyplot as plt

portfolio = pd.read_csv("portfolio.csv")
prices = pd.read_csv("DailyPrices.csv", index_col=0)
returns = prices.pct_change().dropna()
returns = returns - np.mean(returns)

def portfolio_price(dailyprice, portfolio):
    return [dailyprice.iloc[-1][stock] for stock in portfolio['Stock']]

def get_return(portfolio):
    return returns.loc[:, portfolio['Stock']].T


portfolio_A = portfolio[portfolio['Portfolio']=='A']
portfolio_B = portfolio[portfolio['Portfolio']=='B']
portfolio_C = portfolio[portfolio['Portfolio']=='C']
A_return = get_return(portfolio_A)
B_return = get_return(portfolio_B)
C_return = get_return(portfolio_C)
returns = get_return(portfolio)
A_return = A_return.transpose().set_index([A_return.columns.tolist()])
B_return = B_return.transpose().set_index([B_return.columns.tolist()])
C_return = C_return.transpose().set_index([C_return.columns.tolist()])
returns = returns.transpose().set_index([returns.columns.tolist()])

def PCA_with_percent(cov, percent=0.95, num_of_simulation=30000):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[::-1].cumsum()[::-1] / total_variance
    num_components = np.searchsorted(explained_variance_ratio, percent, side='right')

    if num_components == 0:
        raise ValueError("The explained variance ratio is too low.")

    selected_eigenvalues = eigenvalues[-num_components:]
    selected_eigenvectors = eigenvectors[:, -num_components:]

    simulate = np.random.normal(size=(num_components, num_of_simulation))

    return selected_eigenvectors @ np.diag(np.sqrt(selected_eigenvalues)) @ simulate

def portValues(portfolio, returns_portfolio, price):
    # Calculate parameters and CDFs for each stock in the portfolio
    parameters = []
    return_cdfs = []
    for col in returns_portfolio.columns:
        params = t.fit(returns_portfolio[col].values)
        parameters.append(params)
        cdf = t.cdf(returns_portfolio[col].values, *params)
        return_cdfs.append(cdf.tolist())
    return_cdfs = pd.DataFrame(return_cdfs).T
    
    # Calculate the Spearman correlation matrix and perform PCA
    spearman_cor = return_cdfs.corr(method='spearman')
    pca = PCA_with_percent(spearman_cor)
    pca_df = pd.DataFrame(pca).T
    
    # Calculate the simulated returns for each stock
    simulated_returns = []
    for col, params in zip(pca_df.columns, parameters):
        cdf = norm.cdf(pca_df[col].values, loc=0, scale=1)
        simulated_returns.append(t.ppf(cdf, *params))
    simulated_returns = np.array(simulated_returns)
    
    # Calculate the simulated prices and portfolio values
    simulated_prices = (1 + simulated_returns.T) * price
    portfolio_values = simulated_prices.dot(portfolio['Holding'])
    portfolio_values.sort()
    return portfolio_values

def calculate_var(data, mean=0, alpha=0.05):
    return mean-np.quantile(data, alpha)

def calculate_es(data, mean = 0, alpha=0.05):
    return -np.mean(data[data<-calculate_var(data, mean, alpha)])

A_p = portValues(portfolio_A, A_return, portfolio_price(prices, portfolio_A))
B_p = portValues(portfolio_B, B_return, portfolio_price(prices, portfolio_B))
C_p = portValues(portfolio_C, C_return, portfolio_price(prices, portfolio_C))
p = portValues(portfolio, returns, portfolio_price(prices, portfolio))

# For portfolio A
print("VaR and ES under T distribution for Portfolio A are:")
total_value = np.sum(portfolio_A['Holding']*portfolio_price(prices,portfolio_A))
print("VaR: ", calculate_var(A_p, total_value))
print("ES: ", calculate_es(total_value - A_p))
var = calculate_var(A_p, total_value)
es = calculate_es(total_value - A_p)
plt.hist(A_p - total_value, bins=50, density=True)
plt.axvline(x= -1*var, color='r', linestyle='--', label='VaR')
plt.axvline(x= -1*es, color='g', linestyle='--', label='ES')
plt.legend()
plt.title('Portfolio A Return Distribution with VaR and ES')
plt.xlabel('Portfolio Return')
plt.ylabel('Density')
plt.show()

# For portfolio B
print("VaR and ES under T distribution for Portfolio B are:")
total_value = np.sum(portfolio_B['Holding']*portfolio_price(prices,portfolio_B))
print("VaR: ", calculate_var(B_p, total_value))
print("ES: ", calculate_es(total_value - B_p))
var = calculate_var(B_p, total_value)
es = calculate_es(total_value - B_p)
plt.hist(B_p - total_value, bins=50, density=True)
plt.axvline(x= -1*var, color='r', linestyle='--', label='VaR')
plt.axvline(x= -1*es, color='g', linestyle='--', label='ES')
plt.legend()
plt.title('Portfolio B Return Distribution with VaR and ES')
plt.xlabel('Portfolio Return')
plt.ylabel('Density')
plt.show()

# For portfolio C
print("VaR and ES under T distribution for Portfolio C are:")
total_value = np.sum(portfolio_C['Holding']*portfolio_price(prices,portfolio_C))
print("VaR: ", calculate_var(C_p, total_value))
print("ES: ", calculate_es(total_value - C_p))
var = calculate_var(C_p, total_value)
es = calculate_es(total_value - C_p)
plt.hist(C_p - total_value, bins=50, density=True)
plt.axvline(x= -1*var, color='r', linestyle='--', label='VaR')
plt.axvline(x= -1*es, color='g', linestyle='--', label='ES')
plt.legend()
plt.title('Portfolio C Return Distribution with VaR and ES')
plt.xlabel('Portfolio Return')
plt.ylabel('Density')
plt.show()

# For the total portfolio
print("VaR and ES under T distribution for total portfolio are:")
total_value = np.sum(portfolio['Holding']*portfolio_price(prices,portfolio))
print("VaR: ", calculate_var(p, total_value))
print("ES: ", calculate_es(total_value - p))
var = calculate_var(p, total_value)
es = calculate_es(total_value - p)
plt.hist(p - total_value, bins=50, density=True)
plt.axvline(x= -1*var, color='r', linestyle='--', label='VaR')
plt.axvline(x= -1*es, color='g', linestyle='--', label='ES')
plt.legend()
plt.title('Total Portfolio Return Distribution with VaR and ES')
plt.xlabel('Portfolio Return')
plt.ylabel('Density')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




