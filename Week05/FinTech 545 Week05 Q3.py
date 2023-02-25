#!/usr/bin/env python
# coding: utf-8

# In[137]:


import numpy as np
import pandas as pd
from numpy.linalg import eig
import datetime
from scipy.stats import norm

def fit_gen_t(x):
    # Estimate the parameters of the Generalized T distribution using the method of moments
    mean = np.mean(x)
    var = np.var(x, ddof=1)
    skew = np.mean(((x - mean) / np.sqrt(var)) ** 3)
    kurt = np.mean(((x - mean) / np.sqrt(var)) ** 4) - 3
    nu = kurt / 2 + 3
    lam = skew / np.sqrt((nu - 2) * (nu - 4))

    # Fit the Generalized T distribution to the data
    params = gennorm.fit(x, loc=mean, scale=np.sqrt(var), shape=nu, floc=mean, fscale=np.sqrt(var), loc0=mean, scale0=np.sqrt(var), shape0=nu, solver='lbfgsb')
    return params



def simulate_gen_t(a, nsim, pctExp=1, mean=[], seed=1234):
    n = a.shape[0]

    # If the mean is missing then set to 0, otherwise use provided mean
    _mean = np.zeros(n)
    if len(mean) != 0:
        _mean = mean

    # Eigenvalue decomposition
    vals, vecs = eig(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    # Julia returns values lowest to highest, flip them and the vectors
    flip = np.arange(vals.size - 1, -1, -1)
    vals = vals[flip]
    vecs = vecs[:, flip]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0.0
        # figure out how many factors we need for the requested percent explained
        for i in range(posv.size):
            pct += vals[i]/tv
            nval += 1
            if pct >= pctExp:
                break
        if nval < posv.size:
            posv = posv[:nval]
    vals = vals[posv]

    vecs = vecs[:, posv]

    # print(f"Simulating with {posv.size} PC Factors: {np.sum(vals)/tv*100}% total variance explained")
    B = vecs @ np.diag(np.sqrt(vals))

    np.random.seed(seed)
    m = vals.size
    r = np.empty((m, nsim))
    for i in range(nsim):
        params = [fit_gen_t(x) for x in B.T]
        r[:, i] = gennorm.rvs(*params)

    out = (B @ r).T
    # Loop over iterations and add the mean
    for i in range(n):
        out[:, i] += _mean[i]
    return out


portfolio = pd.read_csv("portfolio.csv")
prices = pd.read_csv("DailyPrices.csv")
returns = return_calculate(prices, dateColumn="Date")

covar = fit_gen_t(np.array(returns[portfolio["Stock"]]))

current = prices.iloc[-1][portfolio["Stock"]]

nSim = 100
sim = simulate_pca(covar, 100)
simReturns = pd.DataFrame(sim, columns=portfolio["Stock"])
iterations = pd.DataFrame({"iteration": [x for x in range(1, nSim+1)]})
values = portfolio.merge(iterations, how='cross')
nVals = values.shape[0]
currentValue = np.empty(nVals, dtype=float)
simulatedValue = np.empty(nVals, dtype=float)
pnl = np.empty(nVals, dtype=float)

for i in range(nVals):
    price = current[values['Stock'][i]]
    currentValue[i] = values['Holding'][i] * price
    simulatedValue[i] = values['Holding'][i] * price * (1.0 + simReturns.loc[values['iteration'][i] % nSim, values['Stock'][i]])
    pnl[i] = simulatedValue[i] - currentValue[i]

values.loc[:, 'currentValue'] = currentValue
values.loc[:, 'simulatedValue'] = simulatedValue
values.loc[:, 'pnl'] = pnl
values

gdf = values.groupby(['Portfolio', 'iteration'])
portfolioValues = gdf.agg({'currentValue': 'sum', 'pnl': 'sum'}).rename(columns={'currentValue': 'currentValue', 'pnl': 'pnl'})

gdf = portfolioValues.groupby(['Portfolio'])
portfolioRisk = gdf.agg({
    'currentValue': lambda x: x.iloc[0],
    'pnl': lambda x: norm.ppf(0.05, x.mean(), x.std())
}).rename(columns={'currentValue': 'currentValue', 'pnl': 'VaR95'})

gdf = values.groupby(['iteration'])
totalValues = values.groupby('iteration').agg({'currentValue': 'sum', 'pnl': 'sum'}).reset_index()

print(totalValues)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




