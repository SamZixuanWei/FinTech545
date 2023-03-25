#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve

# read in the data
df = pd.read_csv('problem3.csv')

# set the input parameters
S = 151.03 # current AAPL price
r = 0.0425 # risk-free rate
q = 0.0053 # continuously compounding coupon
today = '03/03/2023' # current date in mm/dd/yyyy format
today = pd.to_datetime(today, format='%m/%d/%Y')

# define the Black-Scholes formula
def black_scholes(price, strike, t, r, q, option_type, sigma):
    d1 = (np.log(price / strike) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == 'call':
        return price * np.exp(-q * t) * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)
    else:
        return strike * np.exp(-r * t) * norm.cdf(-d2) - price * np.exp(-q * t) * norm.cdf(-d1)

# define the function to solve for implied volatility
def implied_volatility(price, strike, t, r, q, option_type, market_price):
    return fsolve(lambda x: black_scholes(price, strike, t, r, q, option_type, x) - market_price, 0.5)[0]

# calculate the implied volatility for each option
for i, row in df.iterrows():
    if row['Type'] == 'Option':
        price = S
        strike = row['Strike']
        t = (pd.to_datetime(row['ExpirationDate'], format='%m/%d/%Y') - today).days / 365.0
        option_type = row['OptionType'].lower()
        market_price = row['CurrentPrice']
        iv = implied_volatility(price, strike, t, r, q, option_type, market_price)
        print(f"Implied volatility for {option_type.upper()} option in {row['Portfolio']} portfolio is {iv}")

# Straddle
S = np.linspace(100, 200, 101)  # underlying price range
K = 150  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma_call = 0.2677
sigma_put = 0.2597
Price_call = 6.8
Price_put = 4.85

# Calculate the d1 and d2 parameters
d1_call = (np.log(S/K) + (r - q + 0.5*sigma_call**2)*T) / (sigma_call*np.sqrt(T))
d2_call = d1_call - sigma_call*np.sqrt(T)
d1_put = (np.log(S/K) + (r - q + 0.5*sigma_put**2)*T) / (sigma_put*np.sqrt(T))
d2_put = d1_put - sigma_put*np.sqrt(T)
# Calculate the call and put option values using the Black-Scholes formula
call_value = S*norm.cdf(d1_call) - np.exp(-(r-q)*T)*K*norm.cdf(d2_call) - Price_call
put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2_put) - S*norm.cdf(-d1_put) - Price_put


# Calculate the straddle value
straddle_value = call_value + put_value

plt.plot(S, straddle_value)
plt.xlabel('Underlying Price')
plt.ylabel('Straddle Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[11]:


# SynLong
S = np.linspace(100, 200, 101)  # underlying price range
K = 150  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma_call = 0.2677
sigma_put = 0.2597
Price_call = 6.8
Price_put = 4.85

# Calculate the d1 and d2 parameters
d1_call = (np.log(S/K) + (r - q + 0.5*sigma_call**2)*T) / (sigma_call*np.sqrt(T))
d2_call = d1_call - sigma_call*np.sqrt(T)
d1_put = (np.log(S/K) + (r - q + 0.5*sigma_put**2)*T) / (sigma_put*np.sqrt(T))
d2_put = d1_put - sigma_put*np.sqrt(T)

# Calculate the call and put option values using the Black-Scholes formula
call_value = S*norm.cdf(d1_call) - np.exp(-(r-q)*T)*K*norm.cdf(d2_call) - Price_call
put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2_put) - S*norm.cdf(-d1_put) + Price_put

# Calculate the SynLong value
synlong_value = call_value - put_value

plt.plot(S, synlong_value)
plt.xlabel('Underlying Price')
plt.ylabel('SynLong Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[12]:


S = np.linspace(100, 200, 101)  # underlying price range
K = 150  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2677  # volatility
Price_call1 = 6.8
Price_call2 = 2.21

# Calculate the d1 and d2 parameters
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

# Calculate two call options values using the Black-Scholes formula
call_value1 = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) - Price_call1

S = np.linspace(100, 200, 101)  # underlying price range
K = 160  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2352 # volatility
Price_call1 = 6.8
Price_call2 = 2.21

d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

call_value2 = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) + Price_call2

call_spread_value = call_value1 - call_value2

plt.plot(S, call_spread_value)
plt.xlabel('Underlying Price')
plt.ylabel('Call Spread Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[13]:


S = np.linspace(100, 200, 101)  # underlying price range
K = 150  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2597 # volatility
Price_put1 = 4.85
Price_put2 = 1.84

# Calculate the d1 and d2 parameters
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

# Calculate two call options values using the Black-Scholes formula
put_value1 = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1)- Price_put1

S = np.linspace(100, 200, 101)  # underlying price range
K = 140  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2810  # volatility

d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

put_value2 = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1)+ Price_put2

put_spread_value = put_value1 - put_value2

plt.plot(S, put_spread_value)
plt.xlabel('Underlying Price')
plt.ylabel('Put Spread Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[14]:


S = np.linspace(100, 200, 101)
P = 151.03
stock_value = S - P
plt.plot(S, stock_value)
plt.xlabel('Underlying Price')
plt.ylabel('Stock Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[15]:


S = np.linspace(100, 200, 101)  # underlying price range
K = 150  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2677  # volatility
call_price = 6.8

d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

call_value = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) - call_price

plt.plot(S, call_value)
plt.xlabel('Underlying Price')
plt.ylabel('Call Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[16]:


S = np.linspace(100, 200, 101)  # underlying price range
K = 150  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2597  # volatility
put_price = 4.85

d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1) - put_price

plt.plot(S, put_value)
plt.xlabel('Underlying Price')
plt.ylabel('Put Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[17]:


S = np.linspace(100, 200, 101)
P = 151.03
stock_value = S - P

S = np.linspace(100, 200, 101)  # underlying price range
K = 155  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2468  # volatility
Price_call = 4.05

# Calculate the d1 and d2 parameters
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

# Calculate two call options values using the Black-Scholes formula
call_value = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) + Price_call

Covered_call_value = stock_value - call_value

plt.plot(S, Covered_call_value)
plt.xlabel('Underlying Price')
plt.ylabel('Covered call Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[18]:


S = np.linspace(100, 200, 101)
P = 151.03
stock_value = S - P

S = np.linspace(100, 200, 101)  # underlying price range
K = 145  # strike price
T = 0.1521  # time to expiration in years (calculated as days/365)
r = 0.0425  # risk-free rate
q = 0.0053  # continuously compounded dividend yield
sigma = 0.2675  # volatility
Price_put = 3.01

# Calculate the d1 and d2 parameters
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)

# Calculate two call options values using the Black-Scholes formula
put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1) - Price_put

Protected_put_value = stock_value + put_value

plt.plot(S, Protected_put_value)
plt.xlabel('Underlying Price')
plt.ylabel('Protected put Value')
plt.title('Portfolio Value vs Underlying Price')
plt.show()


# In[ ]:





# In[217]:


def simulate_price_with_ar1(current_price, returns, n_steps, n_simulation=10000):
    model = ARIMA(returns, order=(1, 0, 0)).fit()   
    returns = returns.values
    alpha_1 = model.params[1] 
    beta = model.params[0]
    resid = model.resid
    sigma = np.std(resid)    
    simulated_returns = np.zeros((n_steps, n_simulation))
    for i in range(n_simulation): 
        simulated_returns[0, i] = beta + alpha_1 * returns[-1] + sigma * np.random.normal() 
        for j in range(1, n_steps):
            simulated_returns[j, i] = beta + alpha_1 * simulated_returns[j-1, i] + sigma * np.random.normal()
    simulated_prices = current_price * np.exp(simulated_returns.cumsum(axis=0)) 
    return simulated_prices


simulated_prices = pd.DataFrame(simulate_price_with_ar1(current_price, log_returns_demean, 10))

simulated_prices_list = simulated_prices.values.tolist()
simulated_prices_list = simulated_prices_list[1]

straddle_values = []
for prices in simulated_prices_list:
    S = prices  # underlying price range
    K = 150  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma_call = 0.2677
    sigma_put = 0.2597
    Price_call = 6.8
    Price_put = 4.85

    # Calculate the d1 and d2 parameters
    d1_call = (np.log(S/K) + (r - q + 0.5*sigma_call**2)*T) / (sigma_call*np.sqrt(T))
    d2_call = d1_call - sigma_call*np.sqrt(T)
    d1_put = (np.log(S/K) + (r - q + 0.5*sigma_put**2)*T) / (sigma_put*np.sqrt(T))
    d2_put = d1_put - sigma_put*np.sqrt(T)
    # Calculate the call and put option values using the Black-Scholes formula
    call_value = S*norm.cdf(d1_call) - np.exp(-(r-q)*T)*K*norm.cdf(d2_call) - Price_call
    put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2_put) - S*norm.cdf(-d1_put) - Price_put

    # Calculate the straddle value
    straddle_value = call_value + put_value
    straddle_values.append(straddle_value)
mean_straddle_value = np.mean(straddle_values)
print(mean_straddle_value)
straddle_values = np.array(straddle_values)
var_95 = np.percentile(straddle_values, 5)
print("VaR (95% confidence level):", var_95)
es_95 = np.mean(straddle_values[straddle_values <= var_95])
print("ES (95% confidence level):", es_95)


# In[219]:


synlong_values = []
for prices in simulated_prices_list:
    S = prices  # underlying price range
    K = 150  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma_call = 0.2677
    sigma_put = 0.2597
    Price_call = 6.8
    Price_put = 4.85

    # Calculate the d1 and d2 parameters
    d1_call = (np.log(S/K) + (r - q + 0.5*sigma_call**2)*T) / (sigma_call*np.sqrt(T))
    d2_call = d1_call - sigma_call*np.sqrt(T)
    d1_put = (np.log(S/K) + (r - q + 0.5*sigma_put**2)*T) / (sigma_put*np.sqrt(T))
    d2_put = d1_put - sigma_put*np.sqrt(T)

    # Calculate the call and put option values using the Black-Scholes formula
    call_value = S*norm.cdf(d1_call) - np.exp(-(r-q)*T)*K*norm.cdf(d2_call) - Price_call
    put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2_put) - S*norm.cdf(-d1_put) + Price_put

    # Calculate the SynLong value
    synlong_value = call_value - put_value
    synlong_values.append(synlong_value)
    
mean_synlong_value = np.mean(synlong_values) 
print(mean_synlong_value)
synlong_values = np.array(synlong_values)
var_95 = np.percentile(synlong_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(synlong_values[synlong_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[223]:


callspread_values = []
for prices in simulated_prices_list:
    S = prices  # underlying price range
    K = 150  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2677  # volatility
    Price_call1 = 6.8
    Price_call2 = 2.21

    # Calculate the d1 and d2 parameters
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Calculate two call options values using the Black-Scholes formula
    call_value1 = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) - Price_call1

    S = prices  # underlying price range
    K = 160  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2352 # volatility
    Price_call1 = 6.8
    Price_call2 = 2.21

    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    call_value2 = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) + Price_call2

    call_spread_value = call_value1 - call_value2
    callspread_values.append(call_spread_value)
    
mean_callspread_value = np.mean(callspread_values) 
print(mean_callspread_value)
callspread_values = np.array(callspread_values)
var_95 = np.percentile(callspread_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(callspread_values[callspread_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[224]:


putspread_values = []
for prices in simulated_prices_list:
    S = prices  # underlying price range
    K = 150  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2597 # volatility
    Price_put1 = 4.85
    Price_put2 = 1.84

    # Calculate the d1 and d2 parameters
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Calculate two call options values using the Black-Scholes formula
    put_value1 = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1)- Price_put1

    S = prices  # underlying price range
    K = 140  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2810  # volatility

    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    put_value2 = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1)+ Price_put2

    put_spread_value = put_value1 - put_value2
    putspread_values.append(put_spread_value)
    
mean_putspread_value = np.mean(putspread_values) 
print(mean_putspread_value)
putspread_values = np.array(putspread_values)
var_95 = np.percentile(putspread_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(putspread_values[putspread_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[228]:


stock_values = []
for prices in simulated_prices_list:
    S = prices
    P = 151.03
    stock_value = S - P
    stock_values.append(stock_value)
    
mean_stock_value = np.mean(stock_values) 
print(mean_stock_value)
stock_values = np.array(stock_values)
var_95 = np.percentile(stock_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(stock_values[stock_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[229]:


call_values = []
for prices in simulated_prices_list:
    S = prices  # underlying price range
    K = 150  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2677  # volatility
    call_price = 6.8

    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    call_value = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) - call_price
    call_values.append(call_value)
    
mean_call_value = np.mean(call_values) 
print(mean_call_value)
call_values = np.array(call_values)
var_95 = np.percentile(call_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(call_values[call_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[230]:


put_values = []
for prices in simulated_prices_list:
    S = prices  # underlying price range
    K = 150  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2597  # volatility
    put_price = 4.85

    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1) - put_price
    put_values.append(put_value)
    
mean_put_value = np.mean(put_values) 
print(mean_put_value)
put_values = np.array(put_values)
var_95 = np.percentile(put_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(put_values[put_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[231]:


coveredcall_values = []
for prices in simulated_prices_list:
    S = prices
    P = 151.03
    stock_value = S - P

    S = prices  # underlying price range
    K = 155  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2468  # volatility
    Price_call = 4.05

    # Calculate the d1 and d2 parameters
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Calculate two call options values using the Black-Scholes formula
    call_value = S*norm.cdf(d1) - np.exp(-(r-q)*T)*K*norm.cdf(d2) + Price_call

    covered_call_value = stock_value - call_value
    coveredcall_values.append(covered_call_value)
    
mean_coveredcall_value = np.mean(coveredcall_values) 
print(mean_coveredcall_value)
coveredcall_values = np.array(coveredcall_values)
var_95 = np.percentile(coveredcall_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(coveredcall_values[coveredcall_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[232]:


protectedput_values = []
for prices in simulated_prices_list:
    S = prices
    P = 151.03
    stock_value = S - P

    S = prices  # underlying price range
    K = 145  # strike price
    T = 0.1521  # time to expiration in years (calculated as days/365)
    r = 0.0425  # risk-free rate
    q = 0.0053  # continuously compounded dividend yield
    sigma = 0.2675  # volatility
    Price_put = 3.01

    # Calculate the d1 and d2 parameters
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Calculate two call options values using the Black-Scholes formula
    put_value = np.exp(-(r-q)*T)*K*norm.cdf(-d2) - S*norm.cdf(-d1) - Price_put

    protected_put_value = stock_value + put_value
    protectedput_values.append(protected_put_value)
    
mean_protectedput_value = np.mean(protectedput_values) 
print(mean_protectedput_value)
protectedput_values = np.array(protectedput_values)
var_95 = np.percentile(protectedput_values, 5)
print("VaR (95% confidence level):", -var_95)
es_95 = np.mean(protectedput_values[protectedput_values <= var_95])
print("ES (95% confidence level):", -es_95)


# In[ ]:




