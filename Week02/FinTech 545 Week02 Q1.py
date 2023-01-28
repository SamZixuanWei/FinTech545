#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from scipy.stats import norm, stats,kurtosis, ttest_1samp,skew
import matplotlib.pyplot as plt

# Test the kurtosis function for bias in small sample sizes
d = norm(0,1)
samples = 1000
kurts = np.empty(samples)
for i in range(samples):
    kurts[i] = kurtosis(d.rvs(100))
    
#summary statistics
print(np.mean(kurts), np.std(kurts))

t_value, p_value = stats.ttest_1samp(kurts, d.stats(moments='k'))
print("p-value -", p_value)

#check if the function is unbiased based on the threshold of 5%
print("Kurtosis function is biased in Python with a small sample size?",p_value <= 0.05)

# Test the kurtosis function for bias in large sample sizes
d = norm(0,1)
samples = 1000
kurts = np.empty(samples)
for i in range(samples):
    kurts[i] = kurtosis(d.rvs(100000))
    
#summary statistics
print(np.mean(kurts), np.std(kurts))

t_value, p_value = stats.ttest_1samp(kurts, d.stats(moments='k'))
print("p-value -", p_value)

#check if the function is unbiased based on the threshold of 5%
print("Kurtosis function is biased in Python with a large sample size?",p_value <= 0.05)

# Test the skewness function for bias in small sample sizes
d = norm(0,1)
samples = 1000
skews = np.empty(samples)
for i in range(samples):
    skews[i] = skew(d.rvs(100))
    
#summary statistics
print(np.mean(skews), np.std(skews))

t_value, p_value = stats.ttest_1samp(skews, d.stats(moments='s'))
print("p-value -", p_value)

#check if the function is unbiased based on the threshold of 5%
print("Skewness function is biased in Python with a small sample size?",p_value <= 0.05)

# Test the skewness function for bias in large sample sizes
d = norm(0,1)
samples = 1000
skews = np.empty(samples)
for i in range(samples):
    skews[i] = skew(d.rvs(100000))
    
#summary statistics
print(np.mean(skews), np.std(skews))

t_value, p_value = stats.ttest_1samp(skews, d.stats(moments='s'))
print("p-value -", p_value)

#check if the function is unbiased based on the threshold of 5%
print("Skewness function is biased in Python with a large sample size?",p_value <= 0.05)

#compare the graph of normal distribution with sample size of 100 and 100000
sample100 = np.random.normal(size=100)
plt.hist(sample100, bins=10, density=True, color='b', alpha=0.5)
mu, std = np.mean(sample100), np.std(sample100)
x = np.linspace(mu - 3*std, mu + 3*std, 100)
plt.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * std**2) ), linewidth=2, color='r')
plt.xlabel('Sample values')
plt.ylabel('Frequency')
plt.title('Normal Distribution with Sample Size of 100')
plt.show()

sample100000 = np.random.normal(size=100000)
plt.hist(sample100000, bins=10, density=True, color='b', alpha=0.5)
mu, std = np.mean(sample100000), np.std(sample100000)
x = np.linspace(mu - 3*std, mu + 3*std, 100000)
plt.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * std**2) ), linewidth=2, color='r')
plt.xlabel('Sample values')
plt.ylabel('Frequency')
plt.title('Normal Distribution with Sample Size of 100000')
plt.show()


mu = 0  
sigma = 1  
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)  
y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )  

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Probability density')
plt.title('Normal Distribution')
plt.show()


# In[58]:


import pandas as pd
import numpy as np
import sklearn
from scipy.stats import skew,kurtosis,describe
import matplotlib.pyplot as plt
import random
data = pd.read_csv("problem1.csv")

#find the skewness and kurtosis using python's own function
skewness = skew(data)
kurtosis = kurtosis(data)

def myskewness(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    numerator = np.sum((data - mean)**3)
    skewness = numerator/(n*(std_dev**3))
    return skewness

print("skewness based on my function:",myskewness(data)[0],"skewness based on python:",skewness[0])

def mykurtosis(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    kurtosis = (1/n) * np.sum(((data - mean) / std_dev)**4) - 3
    return kurtosis

print("kurtosis based on my function:",mykurtosis(data)[0],"kurtosis based on python:",kurtosis[0])

random_numbers = np.random.random(10)
print("skewness based on my function:",myskewness(random_numbers),"skewness based on python:",skew(random_numbers))
print("kurtosis based on my function:",mykurtosis(random_numbers),"kurtosis based on python:",kurtosis(random_numbers))


# In[ ]:




