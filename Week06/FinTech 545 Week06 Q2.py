#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm


def calculate_price(S0, K, T, r, q, sigma, option='call'): 
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option == 'call':
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

def calculate_implied_volatility(S0, K, T, r, q, price, option='call'):
    def calculate_error(sigma):
        return calculate_price(S0, K, T, r, q, sigma, option) - price
    
    sigma = 0.5
    max_iterations = 1000
    tolerance = 1e-8
    for i in range(max_iterations):
        error = calculate_error(sigma)
        if abs(error) < tolerance:
            break
        derivative = (calculate_error(sigma + 1e-6) - calculate_error(sigma)) / 1e-8
        sigma = sigma - error / derivative
    return sigma

S0 = 151.03
r = 0.0425
q = 0.0053
current_date = dt.datetime(2023,3,3)

aapl_option = pd.read_csv('AAPL_Options.csv')
aapl_option['Expiration'] = pd.to_datetime(aapl_option['Expiration'])


aapl_option['ImpliedVol'] = 0

for i in range(len(aapl_option)):
    
    if aapl_option['Type'][i] == 'Call':
        option_type = 'call'
    else:
        option_type = 'put'
        
    K = aapl_option['Strike'][i]
    
    T = (aapl_option['Expiration'][i] - current_date).days / 365
    
    price = aapl_option['Last Price'][i]
    
    aapl_option['ImpliedVol'][i] = calculate_implied_volatility(S0, K, T, r, q, price, option_type)

print(aapl_option)
    
calls = aapl_option[aapl_option['Type'] == 'Call']
puts = aapl_option[aapl_option['Type'] == 'Put']

plt.plot(calls['Strike'], calls['ImpliedVol'], label='Calls')
plt.plot(puts['Strike'], puts['ImpliedVol'], label='Puts')

plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.legend()
plt.savefig('Implied Volatility vs Strike Price', dpi =500)
plt.show()


# In[ ]:





# In[ ]:




