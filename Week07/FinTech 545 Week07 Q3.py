#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import fsolve, minimize


# In[24]:


ff_factors = pd.read_csv('F-F_Research_Data_Factors_daily.csv', parse_dates=['Date']).set_index('Date')
mm_factors = pd.read_csv('F-F_Momentum_Factor_daily.csv', parse_dates=['Date']).set_index('Date').rename(columns={'Mom   ':  "Mom"})
factor = (ff_factors.join(mm_factors, how='right') / 100).loc['2013-1-31':]
prices = pd.read_csv('DailyPrices.csv', parse_dates=['Date'])
factors = ['Mkt-RF', 'SMB', 'HML', 'Mom']
stock_returns = prices.set_index('Date').pct_change().dropna()
stocks = ['AAPL', 'META', 'UNH', 'MA', 'MSFT', 'NVDA', 'HD', 'PFE', 'AMZN', 'BRK-B', 'PG', 'XOM', 'TSLA', 'JPM', 'V', 'DIS', 'GOOGL', 'JNJ', 'BAC', 'CSCO']
dataset = stock_returns[stocks].join(factor)
subset = dataset.dropna()
X = subset[factors]
X = sm.add_constant(X)

y = subset[stocks].sub(subset['RF'], axis=0)

betas = pd.DataFrame(index=stocks, columns=factors)
alphas = pd.DataFrame(index=stocks, columns=['Alpha'])
for stock in stocks:
    model = sm.OLS(y[stock], X).fit()
    betas.loc[stock] = model.params[factors]
    alphas.loc[stock] = model.params['const']
sub_return = pd.DataFrame(np.dot(factor[factors],betas.T), index=factor.index, columns=betas.index)
merge_return = pd.merge(sub_return,factor['RF'], left_index=True, right_index=True)
daily_expected_returns = merge_return.add(merge_return['RF'],axis=0).drop('RF',axis=1).add(alphas.T.loc['Alpha'], axis=1)
expected_annual_return = ((daily_expected_returns+1).cumprod().tail(1) ** (1/daily_expected_returns.shape[0]) - 1) * 250
expected_annual_return


# In[30]:


covariance_matrix = dataset[stocks].cov() * 250
covariance_matrix


# In[36]:


def super_efficient_portfolio(returns, rf_rate, cov_matrix):
    if len(returns.shape) == 1:
        num_assets = returns.shape[0]
    else:
        num_assets = returns.shape[1]
    def neg_sharpe_ratio(weights):
        port_return = np.sum(returns * weights)
        port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - rf_rate) / port_std_dev
        return -sharpe_ratio    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'ineq', 'fun': lambda w: w}]
    bounds = [(0, 1) for _ in range(num_assets)]   
    init_weights = np.ones(num_assets) / num_assets  
    opt_result = minimize(neg_sharpe_ratio, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)   
    opt_weights = opt_result.x
    opt_port_return = np.sum(returns * opt_weights)
    opt_port_std_dev = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
    opt_sharpe_ratio = (opt_port_return - rf_rate) / opt_port_std_dev
    return opt_weights*100, opt_sharpe_ratio
weights, sharpe_ratio = super_efficient_portfolio(expected_annual_return.values[0], 0.0425, covariance_matrix)
print(sharpe_ratio)
weights = pd.DataFrame(weights, index=expected_annual_return.columns, columns=['weight %']).round(2).T
weights


# In[ ]:




