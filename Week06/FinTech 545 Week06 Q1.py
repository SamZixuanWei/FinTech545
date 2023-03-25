#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Time to Maturity
days = 14
days_per_year = 365
ttm = days/days_per_year
print("Time to Maturity is:",ttm)


# In[9]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def bs_price(S, K, T, r, q, sigma, option_type):
    d1 = (np.log(S/K) + (r - q + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T)* norm.cdf(-d1)
    return price

S = 165
K_call = 145
K_put = 185
T = ttm
r = 0.0425 
c = 0.0053
sigma_range = np.arange(0.1, 0.80, 0.01)
call_values = [bs_price(S, K_call, T, r, c,sigma,"Call") for sigma in sigma_range]
put_values = [bs_price(S, K_put, T, r, c,sigma,"Put") for sigma in sigma_range]
plt.plot(sigma_range, call_values, label='Call Option')
plt.plot(sigma_range, put_values, label='Put Option')
plt.xlabel('Volatility Range')
plt.ylabel('Option Value')
plt.legend()
plt.show()
print(call_values)
print(put_values)


# In[10]:


S = 165
K_call = 185
K_put = 145
T = ttm
r = 0.0425 
c = 0.0053
sigma_range = np.arange(0.1, 0.80, 0.01)
call_values = [bs_price(S, K_call, T, r, c,sigma,"Call") for sigma in sigma_range]
put_values = [bs_price(S, K_put, T, r, c,sigma,"Put") for sigma in sigma_range]
plt.plot(sigma_range, call_values, label='Call Option')
plt.plot(sigma_range, put_values, label='Put Option')
plt.xlabel('Volatility Range')
plt.ylabel('Option Value')
plt.legend()
plt.show()
print(call_values)
print(put_values)


# In[ ]:





# In[ ]:




