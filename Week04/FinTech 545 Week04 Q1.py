#!/usr/bin/env python
# coding: utf-8

# In[123]:


# Define the normal distribution parameters
mu = 0
sigma = 0.2

# Define the current price and time period
Pt_1 = 100
t = 1

# Calculate the expected value and standard deviation of the price one period ahead for Pt = Pt-1 + r
r_mean = mu
r_sd = sigma
Pt_mean = Pt_1 + r_mean
Pt_sd = np.sqrt(sigma**2 + 0)
print(f"For Pt = Pt-1 + r, the expected value of price one period ahead is {Pt_mean:.2f} and the standard deviation is {Pt_sd:.2f}")

# Calculate the expected value and standard deviation of the price one period ahead for Pt = Pt-1*(1+r)
r_mean = mu
Pt_mean = Pt_1 * (1 + r_mean)
Pt_sd = np.sqrt((Pt_1**2) * sigma**2 + (0 * (1 + r_mean)**2))
print(f"For Pt = Pt-1*(1+r), the expected value of price one period ahead is {Pt_mean:.2f} and the standard deviation is {Pt_sd:.2f}")

# Calculate the expected value and standard deviation of the price one period ahead for Pt = Pt-1*(e**rt)
r_mean = mu
Pt_sd = Pt_1 * np.sqrt(np.exp(sigma**2) - 1) * np.exp(mu + (sigma**2 / 2))
print(f"For Pt = Pt-1*(e**rt), the expected value of price one period ahead is {Pt_mean:.2f} and the standard deviation is {Pt_sd:.2f}")


# In[124]:


import numpy as np
import matplotlib.pyplot as plt

# Set the initial price, number of simulations, and volatility
P0 = 100
n_sims = 10000
sigma = 0.2

# Create arrays to store the expected prices and standard deviations for each simulation
exp_price_cbm = np.zeros(n_sims)
std_price_cbm = np.zeros(n_sims)
exp_price_ars = np.zeros(n_sims)
std_price_ars = np.zeros(n_sims)
exp_price_grm = np.zeros(n_sims)
std_price_grm = np.zeros(n_sims)

for i in range(n_sims):
    # Create an array of normally distributed random numbers with mean 0 and standard deviation sigma
    r = np.random.normal(0, sigma, size=10000)
    
    # Calculate the price at time t for each simulation of Pt_b
    Pt_b = P0 + r
    Pt_b_std = np.std(Pt_b, ddof=1)
    
    # Store the expected prices and standard deviations for each simulation of Pt_b
    exp_price_cbm[i] = Pt_b[-1]
    std_price_cbm[i] = Pt_b_std
    
    # Arithmetic return system
    Pt_a = P0*(1+r)
    Pt_a_std = np.std(Pt_a, ddof=1)
    exp_price_ars[i] =  Pt_a[-1]
    std_price_ars[i] = Pt_a_std
    
    # Log return or geometric Brownian motion
    Pt_l = P0*((math.e)**r)
    Pt_l_std = np.std(Pt_l, ddof=1)
    exp_price_grm[i] = Pt_l[-1]
    std_price_grm[i] = Pt_l_std
    
   
 # Plot the histogram of expected prices and standard deviations for Pt_b
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(exp_price_cbm, bins=20)
axs[0].set_title('Histogram of Expected Prices of Classical Brownian Motion')
axs[0].set_xlabel('Price')
axs[0].set_ylabel('Frequency')

axs[1].hist(std_price_cbm, bins=20)
axs[1].set_title('Histogram of Standard Deviations of Classical Brownian Motion')
axs[1].set_xlabel('Standard Deviation')
axs[1].set_ylabel('Frequency')

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(exp_price_ars, bins=20)
axs[0].set_title('Histogram of Expected Prices of Arithmetic Return System')
axs[0].set_xlabel('Price')
axs[0].set_ylabel('Frequency')

axs[1].hist(std_price_ars, bins=20)
axs[1].set_title('Histogram of Standard Deviations of Arithmetic Return System')
axs[1].set_xlabel('Standard Deviation')
axs[1].set_ylabel('Frequency')

plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(exp_price_grm, bins=20)
axs[0].set_title('Histogram of Expected Prices of Geometric Brownian Motion')
axs[0].set_xlabel('Price')
axs[0].set_ylabel('Frequency')

axs[1].hist(std_price_grm, bins=20)
axs[1].set_title('Histogram of Standard Deviations of Geometric Brownian Motion')
axs[1].set_xlabel('Standard Deviation')
axs[1].set_ylabel('Frequency')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




