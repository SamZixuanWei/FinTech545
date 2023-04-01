#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import norm
from scipy.optimize import fsolve, minimize
import inspect

data = pd.read_csv('DailyPrices.csv')
aapl_return = data['AAPL'].pct_change().dropna()
aapl_norm = aapl_return - aapl_return.mean()

aapl_option = pd.read_csv('problem2.csv')
aapl_option['ExpirationDate'] = pd.to_datetime(aapl_option['ExpirationDate'])

def binomial_tree_american_continous(S, K, T, r, q, sigma, N=200, option_type='call'):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp((r-q)*dt)-d)/(u-d)
    pd = 1-pu
    df = np.exp(-r*dt)
    z = 1 if option_type == 'call' else -1
    def nNodeFunc(n):
        return (n+2)*(n+1)//2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
    nNodes = nNodeFunc(N)
    optionValues = np.empty(nNodes, dtype = float)

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i,j)
            price = S*u**i*d**(j-i)
            optionValues[idx] = max(0,z*(price-K))
            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
    return optionValues[0]

def binomial_tree_american_discrete(S, K, r, T, sigma, N, option_type, dividend_dates=None, dividend_amounts=None):
    if dividend_dates is None or dividend_amounts is None or (len(dividend_amounts)==0) or (len(dividend_dates)==0):
        return binomial_tree_american_continous(S, K, T, r, 0, sigma, N, option_type)
    elif dividend_dates[0] > N:
        return binomial_tree_american_continous(S, K, T, r, 0, sigma, N, option_type)

    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(r*dt)-d)/(u-d)
    pd = 1-pu
    df = np.exp(-r*dt)
    z = 1 if option_type == 'call' else -1
    
    def nNodeFunc(n):
        return (n+2)*(n+1)//2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
   
    nDiv = len(dividend_dates)
    nNodes = nNodeFunc(dividend_dates[0])

    optionValues = np.empty(nNodes, dtype = float)

    for j in range(dividend_dates[0],-1,-1):
        for i in range(j,-1,-1):
            idx = idxFunc(i,j)
            price = S*u**i*d**(j-i)       
            
            if j < dividend_dates[0]:
                #times before the dividend working backward induction
                optionValues[idx] = max(0,z*(price-K))
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)])  )
                
            else:
                no_ex= binomial_tree_american_discrete(price-dividend_amounts[0], K, r, T-dividend_dates[0]*dt, sigma, N-dividend_dates[0], option_type, [x- dividend_dates[0] for x in dividend_dates[1:nDiv]], dividend_amounts[1:nDiv] )
                ex =  max(0,z*(price-K))
                optionValues[idx] = max(no_ex,ex)

    return optionValues[0]

def implied_volatility(S, K, r, T, price, N, option, dividend_dates=None, dividend_amounts=None):
    f1 = lambda z: (binomial_tree_american_discrete(S, K, r, T, z, N, option, dividend_dates, dividend_amounts)-price)
    return fsolve(f1, x0 = 0.2)[0]


# In[8]:


def calculate_portfolio_values(portfolios, underlying_value, days_ahead=0):
    portfolio_values = pd.DataFrame(index=portfolios["Portfolio"].unique(), columns=[underlying_value])
    portfolio_values = portfolio_values.fillna(0)
    
    for i, portfolio in portfolios.iterrows():

        if portfolio["Type"] == "Stock":
            asset_value = underlying_value
            
        else:
            K = portfolio["Strike"]
            T = ((portfolio["ExpirationDate"] - current_date).days - days_ahead) / 365
            price = portfolio["CurrentPrice"]
            dividend_dates = [round(((dt.datetime(2023,3,15) - current_date).days - days_ahead) / ((portfolio["ExpirationDate"] - current_date).days - days_ahead) * N)]
            dividend_amounts = [1]
            sigma = portfolio["ImpliedVol"]
            
            asset_values = []
            for underlying_prices in np.atleast_1d(underlying_value):
                option_values = (binomial_tree_american_discrete(underlying_prices, K, r, T, sigma, 50, portfolio.loc['OptionType'].lower(), dividend_dates, dividend_amounts))
                asset_values.append(option_values)
            asset_value = np.array(asset_values)
        
        portfolio_values.loc[portfolio["Portfolio"], :] += portfolio["Holding"] * asset_value
        
    return portfolio_values


# In[40]:


def simulate_prices(daily_returns, current_price, days=1, n_simulation = 1000):

    mu, std = norm.fit(daily_returns)
    simulated_returns = np.random.normal(mu, std, (days, n_simulation))
    simulate_prices = current_price * np.exp(simulated_returns.cumsum(axis=0))
    
    return simulate_prices


# In[41]:


current_date = dt.datetime(2023,3,3)
S = 151.03
r = 0.0425
days_ahead = 0
N = 300

implied_vols = []

for i, portfolio in aapl_option.iterrows():

    if portfolio["Type"] == "Stock":
        implied_vols.append(None)

    else:
        K = portfolio["Strike"]
        T = ((portfolio["ExpirationDate"] - current_date).days - days_ahead) / 365
        price = portfolio["CurrentPrice"]
        dividend_dates = [round(((dt.datetime(2023,3,15) - current_date).days - days_ahead) / ((portfolio["ExpirationDate"] - current_date).days - days_ahead) * N)]
        dividend_amounts = [1]
        sigma = implied_volatility(S, K, r, T,  price, 50, portfolio.loc['OptionType'].lower(), dividend_dates, dividend_amounts)
        implied_vols.append(sigma)

aapl_option["ImpliedVol"] = implied_vols

current_values = calculate_portfolio_values(aapl_option, S, 0)

np.random.seed(200)
underlying_prices = pd.DataFrame(simulate_prices(aapl_norm, S, 10))
simulate_portfolio_values = calculate_portfolio_values(aapl_option, underlying_prices.loc[9:].values[0], 10)


# In[42]:


merged_df = pd.merge(simulate_portfolio_values, current_values, left_index=True, right_index=True)
price_change = merged_df.sub(merged_df[S], axis=0).drop(S, axis=1)
price_change.columns = price_change.columns.str[0]

portfolio_metrics = pd.DataFrame(index=aapl_option["Portfolio"].unique(), columns=["Mean", "VaR", "ES"])
portfolio_metrics = portfolio_metrics.fillna(0)


# In[43]:


def calculate_var(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)

def calculate_es(data, var):
    return -np.mean(data[data <= -var])


# In[44]:


for index, row in price_change.iterrows():
    mean = row.values.mean()
    var = calculate_var(row.values)
    es = calculate_es(row.values, var)
    
    portfolio_metrics.loc[index, "Mean"] = mean
    portfolio_metrics.loc[index, "VaR"] = var
    portfolio_metrics.loc[index, "ES"] = es
    
portfolio_metrics


# In[45]:


def first_order_der(func, x, delta):
    return (func(x * (1 + delta)) - func(x * (1 - delta))) / (2 * x * delta)

# calculate second order derivative
def second_order_der(func, x, delta):
    return (func(x * (1 + delta)) + func(x * (1 - delta)) - 2 * func(x)) / (x * delta) ** 2

def cal_partial_derivative(func, order, arg_name, delta=1e-5):
  # initialize for argument names and order
    arg_names = list(inspect.signature(func).parameters.keys())
    derivative_fs = {1: first_order_der, 2: second_order_der}
    def partial_derivative(*args, **kwargs):
        # parse argument names and order
        args_dict = dict(list(zip(arg_names, args)) + list(kwargs.items()))
        arg_val = args_dict.pop(arg_name)

        def partial_f(x):
            p_kwargs = {arg_name:x, **args_dict}
            return func(**p_kwargs)
        return derivative_fs[order](partial_f, arg_val, delta)
    return partial_derivative

delta_calculator =  cal_partial_derivative(binomial_tree_american_discrete, 1, 'S')


for i in range(len(aapl_option)):
    if aapl_option.loc[i,"Type"] != "Stock":

        current_date = dt.datetime(2023,3,3)
        N=50
        K = aapl_option.loc[i,"Strike"]
        ExpirationDate = aapl_option.loc[i, "ExpirationDate"]
        T = ((ExpirationDate  - current_date).days ) / 365
        sigma = aapl_option.loc[i,"ImpliedVol"]
        option_type = aapl_option.loc[i,"OptionType"]
        dividend_dates = [round(((dt.datetime(2023,3,15) - current_date).days ) / ((ExpirationDate  - current_date).days ) * N)]
        dividend_amounts = [1]

        aapl_option.loc[i, "Delta"] = delta_calculator(S, K, r, T, sigma, N, option_type.lower(), dividend_dates, dividend_amounts) * aapl_option.loc[i,'Holding']
    else:
        aapl_option.loc[i, "Delta"] = 1 * aapl_option.loc[i,'Holding']

        


# In[46]:


delta = aapl_option.groupby("Portfolio")['Delta'].sum().apply(lambda x: -x * (underlying_prices.loc[9:].values[0]-151.03))
delta_df = pd.DataFrame(np.array([delta.values[i].reshape(-1) for i in range(len(delta))]), index=delta.keys(), columns=underlying_prices.loc[9:].values[0])

price_change_delta = price_change.add(delta_df)

portfolio_metrics_1 = pd.DataFrame(index=aapl_option["Portfolio"].unique(), columns=["Mean", "VaR", "ES"])
portfolio_metrics_1 = portfolio_metrics_1.fillna(0)

for index, row in price_change_delta.iterrows():
    mean = row.values.mean()
    var = calculate_var(row.values)
    es = calculate_es(row.values, var)
    
    portfolio_metrics_1.loc[index, "Mean"] = mean
    portfolio_metrics_1.loc[index, "VaR"] = var
    portfolio_metrics_1.loc[index, "ES"] = es
    
portfolio_metrics_1


# In[47]:


portfolio_metrics - portfolio_metrics_1


# In[ ]:





# In[ ]:




