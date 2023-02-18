#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import statsmodels.api as sm
import matplotlib.pyplot as plt


def return_calculate(prices, method="DISCRETE", dateColumn="date"):
    vars = list(prices.columns)
    nVars = len(vars)
    vars = [var for var in vars if var != dateColumn]
    if nVars == len(vars):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars = nVars-1
    
    p = prices[vars].to_numpy()
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.empty((n-1, m))
    
    for i in range(n-1):
        for j in range(m):
            p2[i,j] = p[i+1,j] / p[i,j]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    
    dates = prices.iloc[1:, prices.columns.get_loc(dateColumn)]
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars[i]] = p2[:,i]
    
    return out


# In[57]:


# Load data into a pandas dataframe
df = pd.read_csv("DailyPrices.csv")
# Call return_calculate function to calculate arithmetic returns
returns = return_calculate(df, method="DISCRETE", dateColumn="Date")
# Print the resulting dataframe
means = returns.mean()
returns = returns.sub(means, axis=1)
print(returns['META'])


# In[93]:



confidence_level = 0.95
time_horizon = 1
means = returns.mean()
stds = returns.std()
VaRs = {}
for stock in returns.columns:
    z_score = norm.ppf(1 - confidence_level)
    VaRs[stock] = -(means[stock] * time_horizon + z_score * stds[stock] * np.sqrt(time_horizon))
print(VaRs['META'])


# In[94]:


import pandas as pd
import numpy as np
from scipy.stats import norm

confidence_level = 0.95
time_horizon = 1

# Calculate the Exponentially Weighted standard deviation for each stock
lambda_param = 0.94
stds = returns.ewm(alpha=lambda_param, min_periods=1).std().iloc[-1]

# Calculate the VaR for each stock using a normal distribution with EWMA volatility
VaRs = {}
for stock in returns.columns:
    z_score = norm.ppf(1 - confidence_level)
    VaRs[stock] = -(returns[stock].iloc[-1] * time_horizon + z_score * stds[stock] * np.sqrt(time_horizon))


# In[95]:


import numpy as np
import pandas as pd
from scipy.stats import norm

confidence_level = 0.95
time_horizon = 1

# calculate the mean of the returns
means = returns.mean()

# fit AR(1) models to the returns
ar1_models = {}
for stock in returns.columns:
    model = sm.tsa.AutoReg(returns[stock].dropna(), lags=1, trend='n').fit()
    ar1_models[stock] = model.params[0]

# calculate VaR using AR(1) models
VaRs = {}
for stock in returns.columns:
    model = sm.tsa.AutoReg(returns[stock].dropna(), lags=1, trend='n').fit()
    resid = model.resid
    std_resid = resid.std()
    z_score = norm.ppf(1 - confidence_level)
    VaRs[stock] = -(means[stock] * time_horizon + ar1_models[stock] * resid.iloc[-1] * time_horizon + z_score * std_resid * np.sqrt(time_horizon))
    
print(VaRs['META'])


# In[96]:


import numpy as np
import pandas as pd
from scipy.stats import t

confidence_level = 0.95
time_horizon = 1

# calculate the mean and standard deviation of the returns
means = returns.mean()
stds = returns.std()

# calculate VaR using MLE fitted T distribution
VaRs = {}
for stock in returns.columns:
    dof, loc, scale = t.fit(returns[stock])
    z_score = t.ppf(1 - confidence_level, dof, loc, scale)
    VaRs[stock] = -(means[stock] * time_horizon + z_score * stds[stock] * np.sqrt(time_horizon))
print(VaRs['META'])


# In[97]:


# Set the confidence level and time horizon
confidence_level = 0.95
time_horizon = 1

# Calculate the VaR for each stock using historical simulation
VaRs = {}
for stock in returns.columns:
    sorted_returns = returns[stock].sort_values()
    index = int(np.ceil(sorted_returns.shape[0] * (1 - confidence_level)))
    VaRs[stock] = -sorted_returns.iloc[index] * np.sqrt(time_horizon)

# Print the VaRs
print(VaRs['META'])


# In[ ]:





# In[ ]:




