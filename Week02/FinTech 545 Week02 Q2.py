#!/usr/bin/env python
# coding: utf-8

# In[13]:


import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm


data = pd.read_csv("problem2.csv")
x = data["x"]
y = data["y"]
X = sm.add_constant(x)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
error = y - results.predict(X)
data.head(100)
print(error)
sns.distplot(error)
mu, std = norm.fit(error)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2)
plt.show()


# In[ ]:





# In[ ]:





# In[46]:


def myll_n(s,beta):
    n = len(error)
    e = error
    s2 = s * s
    ll = - n / 2 * np.log(s2 * 2 * np.pi) - np.sum(e**2) / (2 * s2)
    return ll
beta = 0.6052
s = stats.stdev(error)
print(myll_n(s,b))


# In[48]:


from scipy.stats import t
df = t(len(error) - 1)
def myll_t(s,beta):
    n = len(error)
    e = error
    s2 = s * s
    ll = df.logpdf(error).sum()
    return ll
beta = 0.6052
s = stats.stdev(error)
print(myll_t(s,b))


# In[ ]:





# In[49]:


n = 100
k_n = 2
AIC_n = 2*k_n - myll_n(s,b)
BIC_n = k_n*np.log(n)- myll_n(s,b)
k_t = 3
AIC_t = 2*k_t - myll_t(s,b)
BIC_t = k_t*np.log(n)- myll_t(s,b)
print("Normal AIC: ",AIC_n)
print("Normal BIC: ",BIC_n)
print("t AIC: ",AIC_t)
print("t BIC: ",BIC_t)


# In[ ]:




